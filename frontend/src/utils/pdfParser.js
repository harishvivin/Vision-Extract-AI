import * as pdfjsLib from 'pdfjs-dist';

// Configure PDF.js worker URL from CDN for browser compatibility
pdfjsLib.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version}/pdf.worker.min.mjs`;

/**
 * Parse an uploaded PDF File in the browser.
 * Extracts text content, line items, and renders high-res page Data URLs.
 * @param {File} file - The uploaded PDF File object.
 * @returns {Promise<Array<{page_number: number, preview_url: string, text: string, clean_text: string, lines: Array<string>}>>}
 */
export async function parsePdfInBrowser(file) {
  const arrayBuffer = await file.arrayBuffer();
  const pdfDoc = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
  
  const pageRecords = [];

  for (let i = 1; i <= pdfDoc.numPages; i++) {
    const page = await pdfDoc.getPage(i);
    const textContent = await page.getTextContent();
    
    // Extract line text items
    const lines = textContent.items
      .map((item) => item.str)
      .filter((str) => str.trim().length > 0);
    const pageText = lines.join(' ');

    // Render Page to Canvas
    const viewport = page.getViewport({ scale: 1.8 });
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = viewport.width;
    canvas.height = viewport.height;

    await page.render({ canvasContext: ctx, viewport }).promise;
    const pageDataUrl = canvas.toDataURL('image/png');

    pageRecords.push({
      page_number: i,
      preview_url: pageDataUrl,
      text: pageText,
      clean_text: pageText.toLowerCase(),
      lines: lines
    });
  }

  return pageRecords;
}

/**
 * Dynamically crop a bounding box region from a page Data URL with emerald highlight border.
 * @param {string} pageDataUrl - The base64 Data URL of the target page.
 * @param {Array<number>} bbox - Normalized bounding box [x1, y1, x2, y2].
 * @returns {Promise<string>} Data URL of the cropped region with green highlight box.
 */
export async function cropImageRegion(pageDataUrl, bbox = [0.10, 0.20, 0.90, 0.50]) {
  return new Promise((resolve) => {
    const img = new Image();
    img.crossOrigin = 'Anonymous';
    img.onload = () => {
      const W = img.width;
      const H = img.height;

      const [nx1, ny1, nx2, ny2] = bbox;
      let x1 = Math.max(0, Math.floor(nx1 * W) - 15);
      let y1 = Math.max(0, Math.floor(ny1 * H) - 15);
      let x2 = Math.min(W, Math.floor(nx2 * W) + 15);
      let y2 = Math.min(H, Math.floor(ny2 * H) + 15);

      const cropW = Math.max(100, x2 - x1);
      const cropH = Math.max(80, y2 - y1);

      const canvas = document.createElement('canvas');
      canvas.width = cropW;
      canvas.height = cropH;
      const ctx = canvas.getContext('2d');

      // Draw cropped image region
      ctx.drawImage(img, x1, y1, cropW, cropH, 0, 0, cropW, cropH);

      // Draw Emerald Green Highlight Border
      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 6;
      ctx.strokeRect(3, 3, cropW - 6, cropH - 6);

      resolve(canvas.toDataURL('image/png'));
    };
    img.onerror = () => resolve(pageDataUrl);
    img.src = pageDataUrl;
  });
}
