#!/usr/bin/env python3
"""Take screenshots of all 3 dashboard tabs for the lab report."""
import asyncio
from pathlib import Path

SCREENSHOTS = Path(__file__).parent.parent / "docs" / "screenshots"
BASE_URL = "http://localhost:8501"


async def scroll_down(page, amount: int = 400):
    """Scroll the Streamlit main container via mouse wheel."""
    await page.mouse.wheel(0, amount)
    await asyncio.sleep(1.5)


async def scroll_to_top(page):
    await page.mouse.wheel(0, -9999)
    await asyncio.sleep(1)


async def shot(page, filename: str, full_page: bool = False):
    path = str(SCREENSHOTS / filename)
    await page.screenshot(path=path, full_page=full_page)
    print(f"  ✓ {filename}")


async def wait_for_streamlit(page):
    """Wait until the tab bar appears and initial spinners clear."""
    await page.wait_for_selector('[data-baseweb="tab"]', timeout=40000)
    try:
        await page.wait_for_selector(
            '[data-testid="stStatusWidget"]', state="detached", timeout=15000
        )
    except Exception:
        pass
    await asyncio.sleep(4)


async def click_tab(page, name_fragment: str):
    tab = page.locator('[data-baseweb="tab"]').filter(has_text=name_fragment)
    await tab.click()
    await asyncio.sleep(2)


async def main():
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        ctx = await browser.new_context(viewport={"width": 1440, "height": 900})
        page = await ctx.new_page()

        # Focus the page so mouse wheel events register
        await page.goto(BASE_URL, wait_until="domcontentloaded")
        await page.mouse.move(720, 450)

        print(f"Waiting for Streamlit to load …")
        await wait_for_streamlit(page)

        # ── Tab 1: Dataset Exploration ────────────────────────────────────────
        print("\n[Tab 1] Dataset Exploration")
        await scroll_to_top(page)
        await shot(page, "lab06_tab1_overview.png")  # metrics + split chart + class dist

        await scroll_down(page, 500)
        await shot(page, "lab06_tab1_class_dist.png")  # class distribution + bottom of charts

        await scroll_down(page, 600)
        await shot(page, "lab06_tab1_sample.png")  # sample browser

        # ── Tab 2: Error Analysis ─────────────────────────────────────────────
        print("\n[Tab 2] Error Analysis")
        await scroll_to_top(page)
        await click_tab(page, "Error Analysis")
        await asyncio.sleep(3)
        await shot(page, "lab06_tab2_runs.png")  # run selection + training curves

        # Click Run Error Analysis button
        print("  Running batch inference (may take 30-60 s on CPU) …")
        run_btn = page.get_by_text("Run Error Analysis on Test Split")
        await run_btn.click()

        # Wait until confusion matrix figure appears (st.pyplot creates stImage)
        try:
            await page.wait_for_selector(
                '[data-testid="stImage"], canvas, svg.main-svg',
                timeout=180000,
            )
            # Extra buffer for the full render to settle
            await asyncio.sleep(5)
        except Exception:
            print("  WARNING: timeout waiting for inference results, continuing …")
            await asyncio.sleep(10)

        # Scroll to see the inference success message + confusion matrix
        await scroll_to_top(page)
        await asyncio.sleep(1)
        await shot(page, "lab06_tab2_confusion.png")

        await scroll_down(page, 600)
        await shot(page, "lab06_tab2_errors.png")

        await scroll_down(page, 700)
        await shot(page, "lab06_tab2_misclassified.png")

        # ── Tab 3: Prediction & Explainability ────────────────────────────────
        print("\n[Tab 3] Prediction & Explainability")
        await scroll_to_top(page)
        await click_tab(page, "Explainability")
        await asyncio.sleep(6)  # Model load + immediate inference

        await scroll_to_top(page)
        await shot(page, "lab06_tab3_prediction.png")  # prediction + probability dist

        # Scroll to Grad-CAM section then generate
        await scroll_down(page, 600)
        await asyncio.sleep(1)

        print("  Generating Grad-CAM …")
        gradcam_btn = page.get_by_text("Generate Grad-CAM")
        await gradcam_btn.click()
        await asyncio.sleep(12)  # Grad-CAM computation + render

        await shot(page, "lab06_tab3_gradcam.png")

        await browser.close()
        print("\nAll 9 screenshots saved to docs/screenshots/")


if __name__ == "__main__":
    asyncio.run(main())
