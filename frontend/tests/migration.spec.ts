import { test, expect } from "@playwright/test";

const API = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

test.describe("NeuraX migration UI", () => {
  test("backend health is displayed in status bar", async ({ page }) => {
    await page.goto("/chat");
    await expect(
      page.getByText(/Backend ready|Backend unavailable/i),
    ).toBeVisible({ timeout: 20_000 });
  });

  test("LM Studio status is understandable", async ({ page }) => {
    await page.goto("/settings");
    await expect(
      page.getByRole("heading", { name: "Settings", exact: true }),
    ).toBeVisible();
    await expect(
      page.getByText(/LM Studio|model|unreachable|available|degraded/i).first(),
    ).toBeVisible({ timeout: 20_000 });
  });

  test("documents page supports upload UI and empty state", async ({
    page,
  }) => {
    await page.goto("/documents");
    await expect(
      page.getByRole("heading", { name: "Documents", exact: true }),
    ).toBeVisible();
    await expect(page.getByText(/Drag and drop files/i)).toBeVisible();
    await expect(page.getByLabel("Upload documents")).toBeAttached();
  });

  test("upload a text document and show indexing progress", async ({
    page,
  }) => {
    test.setTimeout(180_000);
    await page.goto("/documents");
    const fileInput = page.getByLabel("Upload documents");
    await fileInput.setInputFiles({
      name: "playwright-baseline.txt",
      mimeType: "text/plain",
      buffer: Buffer.from(
        "NeuraX Playwright baseline document about offline secure retrieval and citations.",
      ),
    });
    await expect(page.getByText(/Indexing:/i)).toBeVisible({
      timeout: 30_000,
    });
    // Accept in-progress or terminal job states (embedding init can be slow)
    await expect(
      page
        .getByText(
          /Indexing:\s*(queued|running|completed|failed|cancelled)/i,
        )
        .first(),
    ).toBeVisible({ timeout: 30_000 });
  });

  test("chat page can submit a query", async ({ page }) => {
    test.setTimeout(180_000);
    await page.goto("/chat");
    const box = page.getByLabel("Chat query");
    await box.fill("What is offline retrieval?");
    await page.getByLabel("Send query").click();
    await expect(page.getByText("You", { exact: true })).toBeVisible();
    await expect(page.getByText("NeuraX", { exact: true }).first()).toBeVisible(
      { timeout: 120_000 },
    );
  });

  test("keyboard navigation reaches primary landmarks", async ({ page }) => {
    await page.goto("/chat");
    await page.keyboard.press("Tab");
    const focused = page.locator(":focus");
    await expect(focused).toBeVisible();
  });

  test("responsive widths remain usable", async ({ page }) => {
    for (const width of [375, 768, 1280, 1600]) {
      await page.setViewportSize({ width, height: 900 });
      await page.goto("/chat");
      await expect(
        page.getByRole("heading", { name: "Chat", exact: true }),
      ).toBeVisible();
      await page.goto("/documents");
      await expect(
        page.getByRole("heading", { name: "Documents", exact: true }),
      ).toBeVisible();
    }
  });

  test("refresh does not crash application state", async ({ page }) => {
    await page.goto("/chat");
    await page.reload();
    await expect(
      page.getByRole("heading", { name: "Chat", exact: true }),
    ).toBeVisible();
    await expect(page.getByLabel("Chat query")).toBeVisible();
  });

  test("API errors do not expose stack traces in browser", async ({
    request,
  }) => {
    const res = await request.get(`${API}/api/documents/..%2F..%2Fetc%2Fpasswd`);
    const text = await res.text();
    expect(text).not.toContain("Traceback");
    expect(text).not.toMatch(/File \"/);
  });

  test("sources page loads", async ({ page }) => {
    await page.goto("/sources");
    await expect(
      page.getByRole("heading", { name: "Sources", exact: true }),
    ).toBeVisible();
  });
});
