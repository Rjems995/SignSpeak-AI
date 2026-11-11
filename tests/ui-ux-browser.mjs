import {chromium} from 'playwright';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';

const base=process.env.TEST_BASE_URL||'http://127.0.0.1:5000';
const browser=await chromium.launch({headless:true,executablePath:process.env.CHROME_PATH||'C:/Program Files/Google/Chrome/Application/chrome.exe',args:['--use-fake-device-for-media-stream','--use-fake-ui-for-media-stream']});
const errors=[];
try {
  const page=await browser.newPage({viewport:{width:1440,height:1000}});
  page.on('pageerror',error=>errors.push(error.message));
  await page.goto(base+'/translate',{waitUntil:'networkidle'});
  const text=page.locator('#translation-text');
  await text.fill('ABCD');
  await text.evaluate(el=>el.setSelectionRange(2,2));
  await page.click('#add-space');
  assert.equal(await text.inputValue(),'AB CD');
  await page.click('#delete-letter');
  assert.equal(await text.inputValue(),'ABCD');
  await text.evaluate(el=>el.setSelectionRange(1,3));
  await page.click('#add-space');
  assert.equal(await text.inputValue(),'A D');
  await text.fill('   ');
  assert.equal(await page.locator('#delete-letter').isEnabled(),true);
  await page.click('#delete-letter');
  assert.equal(await text.inputValue(),'  ');
  await text.fill('Hello');
  assert.equal(await page.locator('#copy-text').isEnabled(),true);
  assert.equal(await page.locator('#save-text').isDisabled(),true);
  assert.match(await page.locator('#save-hint').innerText(),/Sign in.*copy/i);
  await text.fill('A'.repeat(5000));
  await text.evaluate(el=>el.setSelectionRange(100,101));
  await page.click('#add-space');
  assert.equal((await text.inputValue()).length,5000);
  assert.equal((await text.inputValue())[100],' ');

  // Confidence and progress updates must not repeatedly announce a held letter.
  const announcements=await page.evaluate(async()=>{
    const {CameraController}=await import('/static/js/camera.js');
    const camera=new CameraController();camera.running=true;
    const region=document.getElementById('prediction-announcement');
    let changes=0;const observer=new MutationObserver(records=>changes+=records.length);
    observer.observe(region,{childList:true});
    camera.renderPrediction(true,{label:'A',confidence:.9},{progress:.1});
    await Promise.resolve();
    for(let i=0;i<10;i++)camera.renderPrediction(true,{label:'A',confidence:.91+i/1000},{progress:.2+i/100});
    await Promise.resolve();
    const holding=changes;
    camera.renderPrediction(true,{label:'A',confidence:.95},{progress:1});
    await Promise.resolve();observer.disconnect();
    return {holding,confirmed:changes,text:region.textContent};
  });
  assert.equal(announcements.holding,1);
  assert.equal(announcements.confirmed,2);
  assert.match(announcements.text,/confirmed/);

  await fs.mkdir('artifacts/ui',{recursive:true});
  for(const theme of process.argv.includes('--interactions-only')?[]:['light','dark']) {
    await page.evaluate(theme=>localStorage.setItem('signspeak:settings:guest',JSON.stringify({theme})),theme);
    for(const width of [320,390,768,1024,1440]) {
      await page.setViewportSize({width,height:900});
      for(const route of ['/','/dashboard','/translate','/learn','/practice','/settings','/capture','/login','/register']) {
        await page.goto(base+route,{waitUntil:'networkidle'});
        assert.equal(await page.getAttribute('html','data-theme'),theme);
        assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),`overflow ${theme} ${width} ${route}`);
        const cramped=await page.locator('.button,.icon-button,.alphabet-nav a,.supported-letters a').evaluateAll(elements=>elements.filter(el=>{
          const rect=el.getBoundingClientRect();return rect.width>0&&rect.height>0&&(rect.width<43.5||rect.height<43.5);
        }).map(el=>({id:el.id,text:el.textContent.trim(),width:el.getBoundingClientRect().width,height:el.getBoundingClientRect().height})));
        assert.deepEqual(cramped,[],`small controls ${theme} ${width} ${route}`);
      }
      console.log(`Layout checks passed: ${theme}, ${width}px.`);
    }
    await page.setViewportSize({width:1440,height:1000});
    await page.goto(base+'/translate',{waitUntil:'networkidle'});
    await page.screenshot({path:`artifacts/ui/translator-${theme}-desktop.png`,fullPage:true});
    await page.setViewportSize({width:390,height:844});
    await page.screenshot({path:`artifacts/ui/translator-${theme}-mobile.png`,fullPage:true});
  }
  await page.goto(base+'/learn#letter-R',{waitUntil:'networkidle'});
  assert.equal(await page.locator('.alphabet-nav a[href="#letter-R"]').getAttribute('aria-current'),'location');

  const denied=await browser.newPage({viewport:{width:390,height:844}});
  denied.on('pageerror',error=>errors.push(error.message));
  await denied.addInitScript(()=>{
    let attempts=0;
    navigator.mediaDevices.getUserMedia=()=>new Promise((resolve,reject)=>setTimeout(()=>reject(new DOMException('Mock camera failure',++attempts===1?'NotAllowedError':'NotFoundError')),150));
  });
  await denied.goto(base+'/translate',{waitUntil:'networkidle'});
  await denied.click('#camera-start');
  await denied.waitForFunction(()=>!document.getElementById('camera-error').hidden);
  assert.match(await denied.locator('#camera-error').innerText(),/permission was denied/);
  assert.equal(await denied.locator('#toast').isVisible(),false,'inline errors must not obscure the camera with a duplicate toast');
  assert.equal(await denied.locator('#prediction-label').innerText(),'Camera off');
  await denied.waitForTimeout(5200);
  assert.equal(await denied.locator('#toast').isVisible(),false);
  assert.equal(await denied.locator('#camera-error').isVisible(),true);
  await denied.click('#camera-start');
  await denied.waitForFunction(()=>document.getElementById('camera-start').getAttribute('aria-busy')==='true'&&document.getElementById('camera-error').hidden);
  await denied.waitForFunction(()=>document.getElementById('camera-error').textContent.includes('No camera was found'));
  assert.equal(await denied.locator('#camera-start').isEnabled(),true);
  await denied.screenshot({path:'artifacts/ui/camera-error-mobile.png',fullPage:true});
  assert.deepEqual(errors,[]);
  console.log('UI interaction checks passed: cursor editing, guest save guidance, quiet announcements and persistent camera errors/retry.');
  if(!process.argv.includes('--interactions-only'))console.log('Responsive checks passed: 320–1440px layouts, light/dark themes and 44px controls.');
} finally {await browser.close();}
