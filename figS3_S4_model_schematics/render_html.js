const { chromium } = require('playwright');
(async () => {
  const path = process.argv[2];
  const outPath = process.argv[3];
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto('file://' + path);
  await page.waitForTimeout(200);
  const body = await page.$('body');
  const box = await body.boundingBox();
  await page.pdf({
    path: outPath,
    width: Math.ceil(box.width) + 'px',
    height: Math.ceil(box.height) + 'px',
    printBackground: true,
    margin: {top:0,bottom:0,left:0,right:0}
  });
  await browser.close();
})();
