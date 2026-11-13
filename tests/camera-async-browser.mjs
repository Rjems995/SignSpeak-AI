// Camera lifecycle checks use deferred tracker/frame fixtures, never a webcam
// or a real classifier. Recorded-gesture/classifier parity has separate suites.
import { chromium } from 'playwright';
import assert from 'node:assert/strict';

const base = process.env.TEST_BASE_URL || 'http://127.0.0.1:5000';
const browser = await chromium.launch({
  headless: true,
  executablePath:
    process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe',
  args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream'],
});
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } }),
    errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.route('https://fonts.googleapis.com/**', (route) => route.abort());
  await page.goto(base + '/translate', { waitUntil: 'networkidle' });
  const checks = await page.evaluate(async () => {
    const { CameraController } = await import('/static/js/camera.js');
    const expect = (condition, message) => {
      if (!condition) throw new Error(message);
    };
    const deferred = () => {
      let resolve, reject;
      const promise = new Promise((yes, no) => {
        resolve = yes;
        reject = no;
      });
      return { promise, resolve, reject };
    };
    const flush = async () => {
      for (let i = 0; i < 5; i++) await Promise.resolve();
    };
    const points = Array.from({ length: 21 }, (_, i) => ({
      x: 0.2 + i * 0.01,
      y: 0.3 + i * 0.01,
      z: 0,
    }));
    const detected = { landmarks: [points], worldLandmarks: [points] };
    const empty = { landmarks: [] };
    const results = [];
    function make(options = {}) {
      const camera = new CameraController({
        captureFrame: async () => ({ close() {} }),
        ...options,
      });
      camera.video = {
        readyState: 2,
        currentTime: 1,
        videoWidth: 640,
        videoHeight: 480,
        srcObject: null,
        pause() {},
      };
      camera.running = true;
      camera.lastVideoTime = -1;
      camera.lastFrame = -Infinity;
      camera.classifier = { predict: () => ({ label: null, confidence: 0 }) };
      camera.tracker = {
        detect: async (bitmap) => {
          bitmap.close();
          return empty;
        },
        close() {},
      };
      camera.clearError();
      return camera;
    }
    function frame(camera) {
      camera.lastFrame = -Infinity;
      camera.video.currentTime++;
      const pending = camera.frame();
      // Prevent a synthetic fixture from creating a second uncontrolled RAF
      // loop while its deliberately slow capture or inference is pending.
      cancelAnimationFrame(camera.raf);
      return pending;
    }

    {
      const captured = deferred(),
        inferred = deferred();
      let captures = 0,
        detections = 0,
        closed = 0;
      const camera = make({
        captureFrame: () => {
          captures++;
          return captured.promise;
        },
      });
      camera.tracker = {
        detect: (bitmap) => {
          detections++;
          bitmap.close();
          return inferred.promise;
        },
        close() {},
      };
      const first = frame(camera);
      await frame(camera);
      await frame(camera);
      expect(
        captures === 1 && detections === 0,
        'busy capture must not queue additional camera frames',
      );
      captured.resolve({
        close() {
          closed++;
        },
      });
      await flush();
      expect(detections === 1, 'one captured frame must submit exactly one inference');
      await frame(camera);
      await frame(camera);
      expect(
        captures === 1 && detections === 1,
        'busy inference must not capture or queue more frames',
      );
      inferred.resolve(empty);
      await first;
      await flush();
      expect(
        camera.pendingFrame === null && captures === 1 && detections === 1,
        'finishing inference must not drain a hidden frame queue',
      );
      expect(closed === 1, 'the transferred-frame owner must release the fixture');
      camera.stop();
      results.push('one capture/inference in flight; no frame queue');
    }

    {
      const captured = deferred();
      let detections = 0,
        closed = 0,
        trackerClosed = 0;
      const camera = make({ captureFrame: () => captured.promise });
      camera.tracker = {
        detect: async () => {
          detections++;
          return detected;
        },
        close() {
          trackerClosed++;
        },
      };
      const pending = frame(camera);
      camera.stop();
      captured.resolve({
        close() {
          closed++;
        },
      });
      await pending;
      expect(
        detections === 0 && closed === 1,
        'Stop before capture completes must close its bitmap without detection',
      );
      expect(
        trackerClosed === 1 && camera.pendingFrame === null && !camera.running,
        'Stop must release its tracker and pending state',
      );
      results.push('Stop during capture releases the bitmap');
    }

    {
      const oldResult = deferred(),
        newResult = deferred();
      let processed = 0;
      const camera = make({
        onResult: (prediction, hand) => {
          if (hand) processed++;
        },
      });
      camera.tracker = {
        detect: (bitmap) => {
          bitmap.close();
          return oldResult.promise;
        },
        close() {},
      };
      const oldFrame = frame(camera);
      await flush();
      camera.stop();
      // Resume a new synthetic session without permission prompts or ML loads.
      camera.running = true;
      camera.lastVideoTime = -1;
      camera.tracker = {
        detect: (bitmap) => {
          bitmap.close();
          return newResult.promise;
        },
        close() {},
      };
      const currentFrame = frame(camera);
      await flush();
      const currentJob = camera.pendingFrame;
      oldResult.resolve(detected);
      await oldFrame;
      expect(processed === 0, 'a late result from a stopped camera must not update recognition');
      expect(
        camera.pendingFrame === currentJob,
        'an old finally block must not clear the new session job',
      );
      newResult.resolve(detected);
      await currentFrame;
      expect(
        processed === 1 && camera.pendingFrame === null,
        'the new session must still process its own result',
      );
      camera.stop();
      results.push('late old-session results and cleanup are isolated');
    }

    {
      let trackerClosed = 0;
      const camera = make();
      camera.tracker = {
        detect: async (bitmap) => {
          bitmap.close();
          throw new Error('Fixture worker crashed');
        },
        close() {
          trackerClosed++;
        },
      };
      await frame(camera);
      const error = document.getElementById('camera-error');
      expect(
        !camera.running && camera.tracker === null && trackerClosed === 1,
        'worker failure must stop and release the camera session',
      );
      expect(
        !error.hidden && error.textContent.includes('Select Enable camera to retry'),
        'worker failure must leave an actionable inline error',
      );
      expect(
        document.getElementById('camera-status').textContent.includes('Tracking stopped'),
        'worker failure must expose a stopped status',
      );
      expect(!document.getElementById('camera-start').disabled, 'worker failure must allow retry');
      results.push('worker failure stops capture and leaves retry guidance');
    }

    {
      const inferred = deferred(),
        ended = [],
        commits = [];
      let classified = 0;
      const camera = make({
        onCommit: (letter) => commits.push(letter),
        onSequenceEnd: (value) => ended.push(value.cancelled),
      });
      camera.classifier = {
        predict: () => {
          classified++;
          return { label: null, confidence: 0 };
        },
      };
      camera.tracker = {
        detect: (bitmap) => {
          bitmap.close();
          return inferred.promise;
        },
        close() {},
      };
      camera.points = points;
      camera.lastSeen = performance.now();
      camera.motionClassifier = { predict: () => ({ label: 'J', confidence: 0.99 }) };
      const pending = frame(camera);
      await flush();
      expect(
        camera.recordMotion() === true,
        'recording fixture must start with a recent visible hand',
      );
      inferred.resolve(detected);
      await pending;
      expect(
        camera.motionFrames.length === 0 && classified === 0,
        'a frame captured before Record must not enter the motion sequence or static classifier',
      );
      camera.stop();
      expect(
        commits.length === 0 && ended.length === 1 && ended[0] === true,
        'discarded pre-record frames must not create a partial sequence',
      );
      results.push('recording epochs discard pending pre-record frames');
    }

    {
      const inferred = deferred(),
        commits = [],
        sequences = [],
        ended = [];
      let capturedAt,
        predictions = 0;
      const camera = make({
        onCommit: (letter) => commits.push(letter),
        onSequence: (frames) => sequences.push(frames),
        onSequenceEnd: (value) => ended.push(value.cancelled),
      });
      camera.tracker = {
        detect: (bitmap, timestamp) => {
          bitmap.close();
          capturedAt = timestamp;
          return inferred.promise;
        },
        close() {},
      };
      camera.points = points;
      camera.lastSeen = performance.now();
      camera.motionClassifier = {
        predict: (frames) => {
          predictions++;
          expect(frames.length === 1, 'Finish must include the last in-flight recording frame');
          return { label: 'J', confidence: 0.99 };
        },
      };
      expect(camera.recordMotion() === true, 'pending-finish fixture must start recording');
      const started = camera.motionStart,
        pending = frame(camera);
      await flush();
      const finishing = camera.finishMotion();
      camera.finishMotion();
      expect(
        camera.finishingMotion && camera.recording && predictions === 0,
        'Finish must await the captured recording frame instead of classifying a partial sequence',
      );
      await new Promise((resolve) => setTimeout(resolve, 45));
      inferred.resolve(detected);
      await pending;
      await finishing;
      expect(
        sequences.length === 1 && sequences[0].length === 1,
        'the completed motion sequence must be emitted once',
      );
      expect(
        Math.abs(sequences[0][0].t - (capturedAt - started)) < 0.0001,
        'motion timestamps must use frame capture time, not worker completion time',
      );
      expect(
        performance.now() - started - sequences[0][0].t >= 30,
        'the timestamp test must include real asynchronous detection latency',
      );
      camera.finishMotion();
      expect(
        predictions === 1 && commits.join('') === 'J' && ended.length === 1 && ended[0] === false,
        'repeated Finish clicks must commit the completed gesture exactly once',
      );
      camera.stop();
      results.push('Finish awaits the final frame with capture timestamps and one commit');
    }

    {
      const inferred = deferred(),
        commits = [],
        sequences = [],
        ended = [];
      let predictions = 0;
      const camera = make({
        onCommit: (letter) => commits.push(letter),
        onSequence: (frames) => sequences.push(frames),
        onSequenceEnd: (value) => ended.push(value.cancelled),
      });
      camera.tracker = {
        detect: (bitmap) => {
          bitmap.close();
          return inferred.promise;
        },
        close() {},
      };
      camera.points = points;
      camera.lastSeen = performance.now();
      camera.motionClassifier = {
        predict: () => {
          predictions++;
          return { label: 'J', confidence: 0.99 };
        },
      };
      camera.recordMotion();
      const pending = frame(camera);
      await flush();
      const finishing = camera.finishMotion();
      camera.stop();
      inferred.resolve(detected);
      await pending;
      await finishing;
      expect(
        predictions === 0 && commits.length === 0 && sequences.length === 0,
        'Stop during pending Finish must not classify, emit or commit a partial gesture',
      );
      expect(
        ended.length === 1 && ended[0] === true && !camera.recording && !camera.finishingMotion,
        'pending Finish must end as one cancellation',
      );
      results.push('Stop cancels pending Finish without a late gesture');
    }

    {
      const inferred = deferred(),
        commits = [],
        sequences = [],
        ended = [];
      let predictions = 0;
      const camera = make({
        onCommit: (letter) => commits.push(letter),
        onSequence: (frames) => sequences.push(frames),
        onSequenceEnd: (value) => ended.push(value.cancelled),
      });
      camera.tracker = {
        detect: (bitmap) => {
          bitmap.close();
          return inferred.promise;
        },
        close() {},
      };
      camera.points = points;
      camera.lastSeen = performance.now();
      camera.motionClassifier = {
        predict: () => {
          predictions++;
          return { label: 'J', confidence: 0.99 };
        },
      };
      camera.recordMotion();
      const pending = frame(camera);
      await flush();
      const finishing = camera.finishMotion();
      camera.clearMotion();
      inferred.resolve(detected);
      await pending;
      await finishing;
      expect(
        predictions === 0 && commits.length === 0 && sequences.length === 0,
        'clearing a practice target during pending Finish must discard its late gesture',
      );
      expect(
        ended.length === 1 && ended[0] === true,
        'clearing a pending recording must report exactly one cancellation',
      );
      expect(
        camera.running && !camera.recording && !camera.finishingMotion,
        'clearMotion must release the finishing state without stopping the camera',
      );
      expect(
        !document.getElementById('record-motion').disabled &&
          document.getElementById('record-motion').textContent === 'Record J / Z',
        'clearMotion must restore the recording control',
      );
      camera.lastSeen = performance.now();
      expect(
        camera.recordMotion() === true && camera.recording,
        'the next practice target must be able to record another gesture',
      );
      camera.stop();
      results.push(
        'clearing a practice target cancels pending Finish and allows another recording',
      );
    }
    return results;
  });
  assert.equal(checks.length, 8);
  assert.deepEqual(errors, []);
  console.log(
    'Async camera checks passed: ' + checks.join('; ') + '. No real camera was activated.',
  );
} finally {
  await browser.close();
}
