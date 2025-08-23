import argparse
import asyncio
import hashlib
import logging
import os
import time
from dataclasses import dataclass
from typing import List

import aiofiles
import aiofiles.os
import coloredlogs
import httpx
import uvloop
from openai import AsyncOpenAI
from pypdf import PdfReader

from olmocr.check import check_poppler_version
from olmocr.pipeline import PageResult, build_dolma_document, build_page_query
from olmocr.prompts import PageResponse
from olmocr.prompts.anchor import get_anchor_text
from olmocr.train.dataloader import FrontMatterParser

# Setup colored logging
coloredlogs.install(level="INFO", fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")

logger = logging.getLogger(__name__)

# Suppress noisy logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)


@dataclass
class PageTask:
    """A single page to be processed."""

    pdf_path: str
    page_num: int
    local_pdf_path: str
    attempt: int = 0


@dataclass
class CompletedPage:
    """A completed page result."""

    pdf_path: str
    page_num: int
    result: PageResult


@dataclass
class PipelineConfig:
    """Simple, validated configuration."""

    workspace_path: str
    server_url: str
    num_workers: int = 20
    pages_per_worker: int = 10
    max_retries: int = 5
    max_error_rate: float = 0.004
    target_image_dim: int = 1288

    def __post_init__(self):
        if not os.path.isabs(self.workspace_path):
            raise ValueError("workspace_path must be absolute")
        if not self.server_url:
            raise ValueError("server_url is required")


class PageProcessor:
    """Processes individual pages with retries and error handling."""

    def __init__(self, config: PipelineConfig, semaphore: asyncio.Semaphore):
        self.config = config
        self.semaphore = semaphore
        self.client = get_openai_client_async(
            base_url=f"{config.server_url.rstrip('/')}/v1",
            api_key="dummy-key",
        )

    async def check_server_health(self) -> str:
        """Check server connection and return available model name."""
        try:
            models = await self.client.models.list()
            if not models.data:
                raise ValueError("No models available on server")

            model_name = models.data[0].id
            logger.info(f"Server health check passed - using model: {model_name}")
            return model_name

        except Exception as e:
            raise ConnectionError(f"Server health check failed: {e}") from e

    async def process_page(self, task: PageTask) -> PageResult:
        """Process a page with rotation and retry logic."""
        cumulative_rotation = 0

        for attempt in range(self.config.max_retries):
            try:
                result, rotation_correction = await self._attempt_page_processing(task, attempt, cumulative_rotation)
                if result:
                    return result

                # Handle rotation correction
                if rotation_correction is not None:
                    cumulative_rotation = (cumulative_rotation + rotation_correction) % 360
                    logger.info(f"Retrying {task.pdf_path}:{task.page_num} with {rotation_correction}° rotation")

            except (ConnectionError, OSError, asyncio.TimeoutError) as e:
                logger.warning(f"Connection error {task.pdf_path}:{task.page_num} attempt {attempt}: {e}")
                await asyncio.sleep(min(2**attempt, 30))

            except ValueError as e:
                logger.warning(f"Processing error {task.pdf_path}:{task.page_num} attempt {attempt}: {e}")
                continue

            except Exception as e:
                logger.error(f"Unexpected error {task.pdf_path}:{task.page_num} attempt {attempt}: {e}")
                continue

        # Create fallback
        logger.error(f"Creating fallback for {task.pdf_path}:{task.page_num} after {self.config.max_retries} attempts")
        return self._create_fallback_result(task)

    async def _attempt_page_processing(self, task: PageTask, attempt: int, cumulative_rotation: int) -> tuple[PageResult | None, int | None]:
        """Attempt to process a single page. Returns (result, rotation_correction)."""
        async with self.semaphore:
            # Build query
            query = await build_page_query(task.local_pdf_path, task.page_num, self.config.target_image_dim, cumulative_rotation)

            # Make request with low temperature for consistent OCR
            response = await self.client.chat.completions.create(
                model=query["model"], messages=query["messages"], temperature=0.1, max_tokens=query.get("max_tokens")
            )

            # Validate response
            if response.usage and response.usage.total_tokens and response.usage.total_tokens > 16384:
                raise ValueError("Response exceeded context limit")

            if response.choices[0].finish_reason != "stop":
                raise ValueError("Incomplete response")

            # Parse response
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response content")

            parser = FrontMatterParser(front_matter_class=PageResponse)
            front_matter, text = parser._extract_front_matter_and_text(content)
            page_response = parser._parse_front_matter(front_matter, text)

            # Handle rotation
            if not page_response.is_rotation_valid and attempt < self.config.max_retries - 1:
                return None, page_response.rotation_correction  # Signal retry needed with rotation

            # Success
            input_tokens = response.usage.prompt_tokens if response.usage and response.usage.prompt_tokens else 0
            output_tokens = response.usage.completion_tokens if response.usage and response.usage.completion_tokens else 0

            result = PageResult(
                task.pdf_path,
                task.page_num,
                page_response,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                is_fallback=False,
            )
            return result, None

    def _create_fallback_result(self, task: PageTask) -> PageResult:
        """Create fallback result using anchor text."""
        return PageResult(
            task.pdf_path,
            task.page_num,
            PageResponse(
                natural_text=get_anchor_text(task.local_pdf_path, task.page_num, pdf_engine="pdftotext"),
                primary_language=None,
                is_rotation_valid=True,
                rotation_correction=0,
                is_table=False,
                is_diagram=False,
            ),
            input_tokens=0,
            output_tokens=0,
            is_fallback=True,
        )


class DocumentAssembler:
    """Assembles completed pages into documents and writes results."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.workspace_path = config.workspace_path

        # Create directories
        self.completed_dir = os.path.join(self.workspace_path, "completed")
        self.results_dir = os.path.join(self.workspace_path, "results")
        self.markdown_dir = os.path.join(self.workspace_path, "markdown")

        for dir_path in [self.completed_dir, self.results_dir, self.markdown_dir]:
            os.makedirs(dir_path, exist_ok=True)

    async def assemble_document(self, pdf_path: str, page_results: List[PageResult]) -> None:
        """Assemble pages into document and write results."""
        try:
            # Check quality
            failed_pages = sum(1 for result in page_results if result.is_fallback)
            if failed_pages / len(page_results) > self.config.max_error_rate:
                logger.warning(f"Rejecting {pdf_path}: {failed_pages}/{len(page_results)} pages failed")
                return

            # Build document
            dolma_doc = build_dolma_document(pdf_path, page_results)
            if not dolma_doc:
                logger.warning(f"Failed to build document for {pdf_path}")
                return

            await self._write_markdown(pdf_path, dolma_doc["text"])

            await self._mark_completed(pdf_path)

            logger.info(f"Assembled {pdf_path}: {len(page_results)} pages, {failed_pages} fallback")

        except Exception as e:
            logger.error(f"Failed to assemble {pdf_path}: {e}")

    async def _write_markdown(self, pdf_path: str, content: str) -> None:
        """Write markdown file."""
        md_filename = os.path.splitext(os.path.basename(pdf_path))[0] + ".md"
        md_path = os.path.join(self.markdown_dir, md_filename)

        async with aiofiles.open(md_path, "w") as f:
            await f.write(content)

    async def _mark_completed(self, pdf_path: str) -> None:
        """Mark PDF as completed."""
        pdf_hash = hashlib.sha1(pdf_path.encode()).hexdigest()
        marker_path = os.path.join(self.completed_dir, f"{pdf_hash}.done")

        async with aiofiles.open(marker_path, "w"):
            pass

    async def is_completed(self, pdf_path: str) -> bool:
        """Check if PDF has been completed."""
        pdf_hash = hashlib.sha1(pdf_path.encode()).hexdigest()
        marker_path = os.path.join(self.completed_dir, f"{pdf_hash}.done")
        return await aiofiles.os.path.exists(marker_path)


class PdfReaderWorker:
    """Async worker that loads PDFs into the page queue and checks for existing outputs."""

    def __init__(self, config: PipelineConfig, page_queue: asyncio.Queue, pdf_info: dict, completed_pages: dict):
        self.config = config
        self.page_queue = page_queue
        self.pdf_info = pdf_info
        self.completed_pages = completed_pages
        self.markdown_dir = os.path.join(config.workspace_path, "markdown")

    async def process_pdfs(self, pdf_paths: List[str]) -> None:
        """Process PDF paths and add page tasks to queue."""
        for pdf_path in pdf_paths:
            if not os.path.exists(pdf_path):
                logger.warning(f"PDF not found: {pdf_path}")
                continue

            # Check if markdown output already exists
            if await self._has_existing_output(pdf_path):
                logger.info(f"Skipping {pdf_path} - markdown already exists")
                continue

            try:
                await self._add_pdf_to_queue(pdf_path)
            except Exception as e:
                logger.error(f"Failed to process {pdf_path}: {e}")

    async def _has_existing_output(self, pdf_path: str) -> bool:
        """Check if markdown file already exists for this PDF."""
        md_filename = os.path.splitext(os.path.basename(pdf_path))[0] + ".md"
        md_path = os.path.join(self.markdown_dir, md_filename)
        return await aiofiles.os.path.exists(md_path)

    async def _add_pdf_to_queue(self, pdf_path: str) -> None:
        """Add a single PDF's pages to the processing queue."""
        # Get page count
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)

        # Store PDF info
        self.pdf_info[pdf_path] = total_pages
        self.completed_pages[pdf_path] = {}

        # Add page tasks to queue
        for page_num in range(1, total_pages + 1):
            task = PageTask(pdf_path, page_num, pdf_path)
            await self.page_queue.put(task)
            await asyncio.sleep(0.01)  # Yield to event loop

        logger.info(f"Added {total_pages} pages for {pdf_path}")


class WorkflowCoordinator:
    """Coordinates the entire pipeline workflow."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.page_processor = PageProcessor(config, asyncio.Semaphore(config.num_workers * config.pages_per_worker))
        self.document_assembler = DocumentAssembler(config)

        # Workflow state
        self.page_queue = asyncio.Queue()
        self.completed_pages = {}  # {pdf_path: {page_num: PageResult}}
        self.pdf_info = {}  # {pdf_path: total_pages}
        self.pdf_loading_complete = asyncio.Event()  # Signal when PDF loading is done

        # Metrics
        self.start_time = time.time()
        self.processed_pages = 0
        self.failed_pages = 0

    async def run(self, pdf_paths: List[str]) -> None:
        """Run the complete workflow."""
        # Start PDF reader worker
        pdf_reader = PdfReaderWorker(self.config, self.page_queue, self.pdf_info, self.completed_pages)
        pdf_reader_task = asyncio.create_task(pdf_reader.process_pdfs(pdf_paths))

        # Start metrics reporter
        metrics_task = asyncio.create_task(self._report_metrics())

        # Start page workers
        worker_tasks = [asyncio.create_task(self._page_worker(worker_id)) for worker_id in range(self.config.num_workers)]

        try:
            # Wait for PDF reader to finish adding all PDFs
            await pdf_reader_task
            self.pdf_loading_complete.set()  # Signal workers that PDF loading is done

            # Check if any pages were added
            if self.page_queue.empty():
                logger.info("No pages to process")
                return

            logger.info(f"PDF reader finished - processing {self.page_queue.qsize()} pages with {self.config.num_workers} workers")

            # Wait for all page workers to finish
            await asyncio.gather(*worker_tasks)
        finally:
            metrics_task.cancel()
            await asyncio.gather(metrics_task, return_exceptions=True)

        # Final report
        elapsed = time.time() - self.start_time
        rate = self.processed_pages / max(elapsed, 1.0)
        logger.info(f"Completed: {self.processed_pages} pages, {self.failed_pages} failed, {rate:.1f} pages/sec")

    async def _page_worker(self, worker_id: int) -> None:
        """Worker that processes pages with flow control."""
        in_flight = set()

        while True:
            # Fill up to target pages in flight
            while len(in_flight) < self.config.pages_per_worker:
                try:
                    task = self.page_queue.get_nowait()
                    processing_task = asyncio.create_task(self._process_and_collect(task))
                    in_flight.add(processing_task)
                except asyncio.QueueEmpty:
                    break

            if not in_flight:
                # No tasks in flight - check if we should wait for more or exit
                if self.pdf_loading_complete.is_set():
                    # PDF loading is done and no more work - exit
                    break
                else:
                    # PDF loading still happening - wait a bit for new tasks
                    await asyncio.sleep(0.1)
                    continue

            # Wait for at least one to complete
            _done, pending = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
            in_flight = pending

        logger.info(f"Worker {worker_id} finished")

    async def _process_and_collect(self, task: PageTask) -> None:
        """Process page and collect result."""
        try:
            # Process page
            result = await self.page_processor.process_page(task)

            # Collect result
            self.completed_pages[task.pdf_path][task.page_num] = result

            if result.is_fallback:
                self.failed_pages += 1
            else:
                self.processed_pages += 1

            # Check if document is complete
            total_pages = self.pdf_info[task.pdf_path]
            if len(self.completed_pages[task.pdf_path]) == total_pages:
                # Assemble document
                page_results = [self.completed_pages[task.pdf_path][page_num] for page_num in range(1, total_pages + 1)]
                await self.document_assembler.assemble_document(task.pdf_path, page_results)

        except Exception as e:
            logger.error(f"Failed to process {task.pdf_path}:{task.page_num}: {e}")

    async def _report_metrics(self) -> None:
        """Report metrics periodically."""
        while True:
            await asyncio.sleep(10)
            elapsed = time.time() - self.start_time
            rate = self.processed_pages / max(elapsed, 1.0)
            queue_size = self.page_queue.qsize()
            logger.info(f"Queue: {queue_size} | Processed: {self.processed_pages} | Failed: {self.failed_pages} | Rate: {rate:.1f} pages/sec")


def create_config_from_args(args) -> PipelineConfig:
    """Create configuration from command line arguments."""
    return PipelineConfig(
        workspace_path=os.path.abspath(args.workspace),
        server_url=args.server,
        num_workers=args.workers,
        pages_per_worker=args.pages_per_worker,
        max_retries=args.max_retries,
        max_error_rate=args.max_error_rate,
        target_image_dim=args.target_image_dim,
    )


def get_openai_client_async(base_url: str | None = None, api_key: str | None = None) -> AsyncOpenAI:
    """Get an OpenAI API client."""

    limits = httpx.Limits(max_connections=10000, max_keepalive_connections=10000)
    timeout = httpx.Timeout(connect=5.0, read=180.0, write=180.0, pool=180.0)
    http_client = httpx.AsyncClient(http2=False, limits=limits, timeout=timeout)

    return AsyncOpenAI(
        timeout=180.0,
        base_url=base_url,
        api_key=api_key,
        http_client=http_client,
    )


async def main():
    parser = argparse.ArgumentParser(description="OlmOCR Pipeline V2 - Clean Flow-Based Architecture")
    parser.add_argument("workspace", help="Workspace directory")
    parser.add_argument("--pdfs", nargs="*", help="PDF files to process")
    parser.add_argument("--server", required=True, help="vLLM server URL")

    # Configuration
    parser.add_argument("--workers", type=int, default=20, help="Number of workers")
    parser.add_argument("--pages-per-worker", type=int, default=10, help="Pages per worker")
    parser.add_argument("--max-retries", type=int, default=5, help="Max retries per page")
    parser.add_argument("--max-error-rate", type=float, default=0.004, help="Max error rate")
    parser.add_argument("--target-image-dim", type=int, default=1288, help="Target image dimension")

    args = parser.parse_args()

    if not args.pdfs:
        logger.error("No PDFs provided")
        return

    # Create and validate config
    config = create_config_from_args(args)

    logger.info(f"Pipeline V2 starting with {config.num_workers} workers")
    logger.info(f"Server: {config.server_url}")
    logger.info(f"Workspace: {config.workspace_path}")
    logger.info(f"Max in-flight pages: {config.pages_per_worker * config.num_workers}")

    # Check dependencies
    check_poppler_version()

    # Health check server connection
    logger.info("Checking server connection...")
    dummy_semaphore = asyncio.Semaphore(1)
    health_checker = PageProcessor(config, dummy_semaphore)

    try:
        await health_checker.check_server_health()
    except ConnectionError as e:
        logger.error(f"Server connection failed: {e}")
        return

    # Run pipeline
    coordinator = WorkflowCoordinator(config)
    await coordinator.run(args.pdfs)

    logger.info("Pipeline V2 completed")


if __name__ == "__main__":
    uvloop.run(main())
