import asyncio
from playwright.async_api import async_playwright

async def verify():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        try:
            await page.goto("http://localhost:8501", timeout=60000)
            await page.wait_for_selector('div[data-testid="stAppViewContainer"]', timeout=30000)
            await page.screenshot(path="final_verification.png")
            print("Screenshot saved to final_verification.png")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(verify())
