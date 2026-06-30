#!/usr/bin/env node

const { chromium } = require("playwright");

function luminance(rgb) {
  const parts = String(rgb).match(/\d+/g) || [];
  const nums = parts.slice(0, 3).map((n) => Number(n) / 255);
  const linear = nums.map((v) => (v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4));
  return 0.2126 * (linear[0] || 0) + 0.7152 * (linear[1] || 0) + 0.0722 * (linear[2] || 0);
}

async function main() {
  const url = process.argv[2];
  if (!url) {
    console.error("Usage: node reference_inspector.js <url>");
    process.exit(1);
  }

  let browser;
  try {
    browser = await chromium.launch({
      channel: "chrome",
      headless: true,
    });
  } catch (_error) {
    browser = await chromium.launch({
      headless: true,
    });
  }

  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
    await page.goto(url, { waitUntil: "networkidle", timeout: 45000 });

    const data = await page.evaluate(() => {
      const pageLuminance = (rgb) => {
        const parts = String(rgb).match(/\d+/g) || [];
        const nums = parts.slice(0, 3).map((n) => Number(n) / 255);
        const linear = nums.map((v) => (v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4));
        return 0.2126 * (linear[0] || 0) + 0.7152 * (linear[1] || 0) + 0.0722 * (linear[2] || 0);
      };
      const clean = (value) => String(value || "").replace(/\s+/g, " ").trim();
      const classTokens = Array.from(document.querySelectorAll("[class]"))
        .flatMap((el) => (el.className || "").split(/\s+/))
        .map((token) => token.trim())
        .filter(Boolean)
        .filter((token, index, array) => array.indexOf(token) === index)
        .slice(0, 80);

      const headings = Array.from(document.querySelectorAll("h1, h2, h3"))
        .map((el) => clean(el.textContent))
        .filter(Boolean)
        .slice(0, 12);

      const selectors = [
        "[class*='hero']",
        "[class*='banner']",
        "[class*='intro']",
        "header",
        "main section",
        "section",
      ];

      let hero = null;
      for (const selector of selectors) {
        const nodes = Array.from(document.querySelectorAll(selector));
        for (const node of nodes) {
          const rect = node.getBoundingClientRect();
          const hasHeadline = !!node.querySelector("h1, h2");
          if (rect.height < 260) continue;
          if (!hasHeadline && selector !== "section") continue;
          const styles = window.getComputedStyle(node);
          hero = {
            selector,
            height: Math.round(rect.height),
            backgroundColor: styles.backgroundColor,
            backgroundImage: styles.backgroundImage !== "none" ? styles.backgroundImage : "",
            color: styles.color,
            textAlign: styles.textAlign,
            className: clean(node.className),
            heading: clean((node.querySelector("h1, h2") || {}).textContent),
          };
          break;
        }
        if (hero) break;
      }

      const bodyStyles = window.getComputedStyle(document.body);
      const bodyBackground = bodyStyles.backgroundColor;
      const bodyColor = bodyStyles.color;
      const bodyFont = bodyStyles.fontFamily;
      const title = document.title || "";
      const description = document.querySelector('meta[name="description"]')?.getAttribute("content") || "";
      const links = Array.from(document.querySelectorAll("a"))
        .map((el) => clean(el.textContent))
        .filter(Boolean)
        .slice(0, 20);
      const sectionCount = document.querySelectorAll("section").length;
      const imageCount = document.querySelectorAll("img").length;
      const snippet = clean(document.body.innerText).slice(0, 4000);

      return {
        title,
        description,
        bodyBackground,
        bodyColor,
        bodyFont,
        bodyIsDark: bodyBackground ? pageLuminance(bodyBackground) < 0.35 : false,
        classTokens,
        headings,
        links,
        sectionCount,
        imageCount,
        hero,
        snippet,
      };
    });

    process.stdout.write(JSON.stringify(data));
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  console.error(error?.stack || String(error));
  process.exit(1);
});
