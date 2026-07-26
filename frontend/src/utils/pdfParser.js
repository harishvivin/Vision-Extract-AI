import * as pdfjsLib from 'pdfjs-dist';

// Configure PDF.js worker URL from CDN for browser compatibility
pdfjsLib.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version}/pdf.worker.min.mjs`;

/**
 * Parse an uploaded PDF File in the browser.
 * Extracts text content, line items with precise bounding boxes, and renders high-res page Data URLs.
 * @param {File} file - The uploaded PDF File object.
 * @returns {Promise<Array<{page_number: number, preview_url: string, text: string, clean_text: string, blocks: Array<{text: string, bbox: Array<number>}>}>>}
 */
export async function parsePdfInBrowser(file) {
  const arrayBuffer = await file.arrayBuffer();
  const pdfDoc = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
  
  const pageRecords = [];

  for (let i = 1; i <= pdfDoc.numPages; i++) {
    const page = await pdfDoc.getPage(i);
    const viewport = page.getViewport({ scale: 1.8 });
    const textContent = await page.getTextContent();
    
    const blocks = [];
    const textItems = textContent.items;

    for (let item of textItems) {
      const str = item.str ? item.str.trim() : '';
      if (!str) continue;

      let bbox = [0.05, 0.05, 0.95, 0.20];
      if (item.transform && item.transform.length >= 6) {
        try {
          const pdfX = item.transform[4];
          const pdfY = item.transform[5];
          const pdfW = item.width || 50;
          const pdfH = item.height || 12;

          const rect = viewport.convertToViewportRectangle([pdfX, pdfY, pdfX + pdfW, pdfY + pdfH]);
          const vx1 = Math.min(rect[0], rect[2]);
          const vy1 = Math.min(rect[1], rect[3]);
          const vx2 = Math.max(rect[0], rect[2]);
          const vy2 = Math.max(rect[1], rect[3]);

          bbox = [
            Math.max(0, Math.min(1, vx1 / viewport.width)),
            Math.max(0, Math.min(1, vy1 / viewport.height)),
            Math.max(0, Math.min(1, vx2 / viewport.width)),
            Math.max(0, Math.min(1, vy2 / viewport.height))
          ];
        } catch (e) {
          bbox = [0.05, 0.05, 0.95, 0.25];
        }
      }

      blocks.push({
        text: item.str,
        clean: item.str.toLowerCase(),
        bbox: bbox
      });
    }

    // Render Page to Canvas
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = viewport.width;
    canvas.height = viewport.height;

    await page.render({ canvasContext: ctx, viewport }).promise;
    const pageDataUrl = canvas.toDataURL('image/png');
    const fullText = blocks.map(b => b.text).join(' ');

    pageRecords.push({
      page_number: i,
      preview_url: pageDataUrl,
      text: fullText,
      clean_text: fullText.toLowerCase(),
      blocks: blocks
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
export async function cropImageRegion(pageDataUrl, bbox = [0.05, 0.05, 0.95, 0.30]) {
  return new Promise((resolve) => {
    const img = new Image();
    img.crossOrigin = 'Anonymous';
    img.onload = () => {
      const W = img.width;
      const H = img.height;

      let [nx1, ny1, nx2, ny2] = bbox;
      
      // Ensure reasonable crop bounds
      if (nx2 - nx1 < 0.20) {
        nx1 = Math.max(0, nx1 - 0.10);
        nx2 = Math.min(1, nx2 + 0.50);
      }
      if (ny2 - ny1 < 0.08) {
        ny1 = Math.max(0, ny1 - 0.04);
        ny2 = Math.min(1, ny2 + 0.12);
      }

      let x1 = Math.max(0, Math.floor(nx1 * W) - 15);
      let y1 = Math.max(0, Math.floor(ny1 * H) - 15);
      let x2 = Math.min(W, Math.floor(nx2 * W) + 15);
      let y2 = Math.min(H, Math.floor(ny2 * H) + 15);

      const cropW = Math.max(120, x2 - x1);
      const cropH = Math.max(60, y2 - y1);

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
