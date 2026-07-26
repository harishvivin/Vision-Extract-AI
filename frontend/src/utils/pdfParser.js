import * as pdfjsLib from 'pdfjs-dist';

// Configure PDF.js worker URL from CDN for browser compatibility
pdfjsLib.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version}/pdf.worker.min.mjs`;

/**
 * Parse an uploaded PDF File in the browser.
 * Groups text items into structured line rows with precise bounding boxes and renders high-res page Data URLs.
 * @param {File} file - The uploaded PDF File object.
 * @returns {Promise<Array<{page_number: number, preview_url: string, text: string, clean_text: string, blocks: Array<{text: string, clean: string, bbox: Array<number>}>}>>}
 */
export async function parsePdfInBrowser(file) {
  const arrayBuffer = await file.arrayBuffer();
  const pdfDoc = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
  
  const pageRecords = [];

  for (let i = 1; i <= pdfDoc.numPages; i++) {
    const page = await pdfDoc.getPage(i);
    // High DPI scale 2.2 for crystal clear sharp text rendering
    const viewport = page.getViewport({ scale: 2.2 });
    const textContent = await page.getTextContent();
    
    const textItems = textContent.items;
    const lineMap = new Map();

    for (let item of textItems) {
      const str = item.str ? item.str.trim() : '';
      if (!str) continue;

      if (item.transform && item.transform.length >= 6) {
        try {
          const pdfX = item.transform[4];
          const pdfY = item.transform[5];
          const pdfW = item.width || 30;
          const pdfH = item.height || 12;

          const rect = viewport.convertToViewportRectangle([pdfX, pdfY, pdfX + pdfW, pdfY + pdfH]);
          const vx1 = Math.min(rect[0], rect[2]);
          const vy1 = Math.min(rect[1], rect[3]);
          const vx2 = Math.max(rect[0], rect[2]);
          const vy2 = Math.max(rect[1], rect[3]);

          // Group items by vertical row line tolerance (~10px)
          const lineKey = Math.round(vy1 / 10) * 10;

          if (!lineMap.has(lineKey)) {
            lineMap.set(lineKey, {
              items: [],
              vx1: vx1,
              vy1: vy1,
              vx2: vx2,
              vy2: vy2
            });
          }

          const lineObj = lineMap.get(lineKey);
          lineObj.items.push(item.str);
          lineObj.vx1 = Math.min(lineObj.vx1, vx1);
          lineObj.vy1 = Math.min(lineObj.vy1, vy1);
          lineObj.vx2 = Math.max(lineObj.vx2, vx2);
          lineObj.vy2 = Math.max(lineObj.vy2, vy2);
        } catch (e) {
          // Ignore transform errors
        }
      }
    }

    const blocks = [];
    for (let [, lineObj] of lineMap.entries()) {
      const lineText = lineObj.items.join(' ').replace(/\s+/g, ' ').trim();
      if (lineText.length < 2) continue;

      const bbox = [
        Math.max(0, Math.min(1, lineObj.vx1 / viewport.width)),
        Math.max(0, Math.min(1, lineObj.vy1 / viewport.height)),
        Math.max(0, Math.min(1, lineObj.vx2 / viewport.width)),
        Math.max(0, Math.min(1, lineObj.vy2 / viewport.height))
      ];

      blocks.push({
        text: lineText,
        clean: lineText.toLowerCase(),
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
    const fullText = blocks.map(b => b.text).join('\n');

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
 * STRICT VALIDATION: If bbox is null/invalid or answer was not localized, returns NULL.
 * @param {string} pageDataUrl - The base64 Data URL of the target page.
 * @param {Array<number>} bbox - Normalized bounding box [x1, y1, x2, y2].
 * @returns {Promise<string|null>} Data URL of the cropped region or null if bounding box is invalid/absent.
 */
export async function cropImageRegion(pageDataUrl, bbox) {
  if (!pageDataUrl || !bbox || !Array.isArray(bbox) || bbox.length < 4) {
    return null;
  }

  let [nx1, ny1, nx2, ny2] = bbox;
  if (nx2 <= nx1 || ny2 <= ny1 || nx1 < 0 || ny1 < 0 || nx2 > 1 || ny2 > 1) {
    return null;
  }

  return new Promise((resolve) => {
    const img = new Image();
    img.crossOrigin = 'Anonymous';
    img.onload = () => {
      const W = img.width;
      const H = img.height;

      // Pinpoint crop coordinates with tight line padding
      let x1 = Math.max(0, Math.floor(nx1 * W) - 25);
      let y1 = Math.max(0, Math.floor(ny1 * H) - 18);
      let x2 = Math.min(W, Math.floor(nx2 * W) + 25);
      let y2 = Math.min(H, Math.floor(ny2 * H) + 18);

      const cropW = Math.max(220, x2 - x1);
      const cropH = Math.max(50, y2 - y1);

      if (cropW <= 0 || cropH <= 0) {
        resolve(null);
        return;
      }

      const canvas = document.createElement('canvas');
      canvas.width = cropW;
      canvas.height = cropH;
      const ctx = canvas.getContext('2d');

      // Fill background white
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, cropW, cropH);

      // Draw cropped line region
      ctx.drawImage(img, x1, y1, cropW, cropH, 0, 0, cropW, cropH);

      // Draw Emerald Green Highlight Border around exact target line row
      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 5;
      ctx.strokeRect(2, 2, cropW - 4, cropH - 4);

      resolve(canvas.toDataURL('image/png'));
    };
    img.onerror = () => resolve(null);
    img.src = pageDataUrl;
  });
}
