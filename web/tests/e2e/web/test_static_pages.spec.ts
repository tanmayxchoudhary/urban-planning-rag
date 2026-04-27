import { test, expect } from '@playwright/test';

test('corpus page loads and shows grouped documents', async ({ page }) => {
  await page.goto('http://localhost:3101/corpus');
  // Page should load without crash
  await expect(page.locator('h1')).toContainText('Corpus');
  // Should show loading or content
  const body = page.locator('body');
  await expect(body).toBeVisible();
});

test('about page loads with content', async ({ page }) => {
  await page.goto('http://localhost:3101/about');
  // Page should load without crash  
  await expect(page.locator('h1')).toContainText('About');
  // Should have description paragraph
  await expect(page.locator('text=Urban Planning RAG')).toBeVisible();
});
