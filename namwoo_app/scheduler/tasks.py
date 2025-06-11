import logging
import time
import traceback # For detailed exception logging
# ADDED IMPORT FOR TYPE HINTING
from typing import Optional
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.jobstores.base import JobLookupError
from flask import Flask # Required for app context

# Import the sync service logic
# Use relative import as services is a sibling package
from ..services import sync_service

logger = logging.getLogger(__name__)
sync_logger = logging.getLogger('sync') # Use the dedicated sync logger

# Global scheduler instance
scheduler = None
SYNC_JOB_ID = 'woocommerce_product_sync'

_sync_running = False # Simple flag to prevent concurrent sync runs

def run_sync_logic(app: Flask, full_resync: bool = False):
    """
    The core logic that executes the sync process.
    Separated to be callable from both the scheduler and CLI.
    Includes a simple concurrency lock.

    Args:
        app: The Flask application instance.
        full_resync: If True, force a full sync. Otherwise, attempt incremental (if implemented).
    """
    global _sync_running
    if _sync_running:
        sync_logger.warning(f"Sync job '{SYNC_JOB_ID}' attempted to start while already running. Skipping.")
        return

    _sync_running = True
    sync_logger.info(f"--- Starting sync job '{SYNC_JOB_ID}' (Full Resync: {full_resync}) ---")
    start_time = time.time()
    try:
        # Ensure execution within Flask app context
        with app.app_context():
            if full_resync:
                processed, added, updated, failed = sync_service.run_full_sync(app)
            else:
                # Attempt incremental sync (currently a placeholder)
                processed, added, updated, failed = sync_service.run_incremental_sync(app)
                if processed == 0 and added == 0 and updated == 0 and failed == 0:
                     sync_logger.info("Incremental sync skipped (not implemented), running full sync instead.")
                     processed, added, updated, failed = sync_service.run_full_sync(app)

            duration = time.time() - start_time
            sync_logger.info(f"--- Finished sync job '{SYNC_JOB_ID}' in {duration:.2f}s ---")
            sync_logger.info(f"Sync Summary - Processed: {processed}, Added: {added}, Updated: {updated}, Failed: {failed}")

    except Exception as e:
        # Log error even if run within app_context
        duration = time.time() - start_time
        sync_logger.error(f"!!! Sync job '{SYNC_JOB_ID}' failed after {duration:.2f}s !!!", exc_info=True)
        sync_logger.error(f"Traceback:\n{traceback.format_exc()}")
    finally:
        _sync_running = False # Release the lock


def scheduled_sync_job(app: Flask):
    """
    Wrapper function specifically designed to be called by APScheduler.
    It ensures the sync runs within the application context.
    """
    sync_logger.info(f"Scheduler triggered for job '{SYNC_JOB_ID}'.")
    # Determine if this scheduled run should be full or incremental based on config/logic
    # For now, let's default scheduled runs to incremental (which falls back to full currently)
    is_full_sync = False # Change this based on your strategy (e.g., full sync once a day?)
    run_sync_logic(app, full_resync=is_full_sync)


# Corrected type hint using Optional
def start_scheduler(app: Flask) -> Optional[BackgroundScheduler]:
    """
    Initializes and starts the APScheduler for background tasks.

    Args:
        app: The Flask application instance.

    Returns:
        The initialized BackgroundScheduler instance or None if disabled/failed.
    """
    global scheduler
    if scheduler and scheduler.running:
        logger.warning("Scheduler is already running.")
        return scheduler

    interval_minutes = app.config.get('SYNC_INTERVAL_MINUTES', 0)
    if interval_minutes <= 0:
        logger.info("Background sync scheduler is disabled via config (SYNC_INTERVAL_MINUTES <= 0).")
        return None

    try:
        logger.info(f"Initializing APScheduler (BackgroundScheduler). Sync interval: {interval_minutes} minutes.")
        scheduler = BackgroundScheduler(daemon=True) # daemon=True allows app to exit even if scheduler thread is running

        # Add the sync job
        scheduler.add_job(
            func=scheduled_sync_job,
            args=[app], # Pass the Flask app instance to the job
            trigger=IntervalTrigger(minutes=interval_minutes),
            id=SYNC_JOB_ID,
            name='WooCommerce Product Sync',
            replace_existing=True, # Replace if job with same ID exists (e.g., on restart)
            misfire_grace_time=300 # Allow job to run up to 5 mins late if scheduler was busy/down
        )

        # Start the scheduler
        scheduler.start()
        logger.info(f"APScheduler started successfully. Job '{SYNC_JOB_ID}' scheduled.")

        # Add shutdown hook for graceful exit
        import atexit
        atexit.register(lambda: stop_scheduler())

        return scheduler

    except Exception as e:
        logger.exception(f"Failed to initialize or start APScheduler: {e}")
        scheduler = None
        return None


def stop_scheduler():
    """Stops the APScheduler gracefully if it is running."""
    global scheduler
    if scheduler and scheduler.running:
        logger.info("Attempting to shut down APScheduler...")
        try:
            # Wait for currently running jobs to complete before shutting down? (default: True)
            scheduler.shutdown()
            logger.info("APScheduler shut down successfully.")
        except Exception as e:
            logger.exception(f"Error shutting down APScheduler: {e}")
    elif scheduler:
        logger.info("APScheduler was initialized but not running.")
    else:
        logger.info("APScheduler was not initialized.")


# Corrected type hint using Optional
def get_scheduler_status() -> dict:
    """Returns the current status of the scheduler and its jobs."""
    status = {"scheduler_running": False, "jobs": []}
    if scheduler and scheduler.running:
        status["scheduler_running"] = True
        try:
            jobs = scheduler.get_jobs()
            for job in jobs:
                status["jobs"].append({
                    "id": job.id,
                    "name": job.name,
                    "next_run_time": str(job.next_run_time) if job.next_run_time else None,
                    "trigger": str(job.trigger)
                })
        except Exception as e:
            logger.error(f"Failed to get job details from scheduler: {e}")
            status["error"] = str(e)
    return status