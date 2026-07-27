/**
 * Generate a valid, lightweight PDF File object directly in the browser
 * for benchmark sample testing without external file dependencies.
 */
export function createSampleMedicalPdfFile() {
  const pdfContent = `%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>
endobj
4 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
5 0 obj
<< /Length 420 >>
stream
BT
/F1 16 Tf
50 740 Td (City Care General Hospital - Laboratory Report) Tj
/F1 12 Tf
0 -30 Td (Patient Name: MANJIT SINGH) Tj
0 -25 Td (Age: 45 Years | Gender: Male) Tj
0 -25 Td (Diagnosis: Normal Health Screening Evaluation) Tj
0 -25 Td (Hemoglobin: 14.5 g/dL) Tj
0 -25 Td (Serum Creatinine: 1.02 mg/dL) Tj
0 -25 Td (HbA1c: 5.7 %) Tj
0 -25 Td (Blood Pressure: 120/80 mmHg) Tj
0 -25 Td (HIV Screening Test: Non-Reactive) Tj
0 -25 Td (ECG Result: Normal Sinus Rhythm) Tj
ET
endstream
endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000236 00000 n 
0000000307 00000 n 
trailer
<< /Size 6 /Root 1 0 R >>
startxref
778
%%EOF`;

  const blob = new Blob([pdfContent], { type: 'application/pdf' });
  return new File([blob], 'Manjit_Singh_Medical_Report.pdf', { type: 'application/pdf' });
}
