// On narrow screens, switch workspaces without scrolling the document.
export function initTranslatorLayout() {
  const tabs = document.querySelector('.workspace-tabs');
  const buttons = [...tabs.querySelectorAll('[data-workspace-view]')];
  const panels = buttons.map((button) =>
    document.getElementById(button.getAttribute('aria-controls')),
  );
  const labels = panels.map((panel) => panel.getAttribute('aria-labelledby'));
  const compact = matchMedia('(max-width: 980px)');
  const textCameraControls = document.getElementById('text-camera-controls');
  const textCameraStatus = document.getElementById('text-camera-status');
  const textCameraStop = document.getElementById('text-camera-stop');
  const cameraStatus = document.getElementById('camera-status');
  const cameraStop = document.getElementById('camera-stop');
  const messagePreview = document.getElementById('camera-message-preview');
  const messageText = document.getElementById('camera-message-text');
  const messageEdit = document.getElementById('camera-message-edit');
  const text = document.getElementById('translation-text');
  const scrollMessageToEnd = () => {
    if (messagePreview.hidden) return;
    messageText.scrollLeft =
      getComputedStyle(messageText).direction === 'rtl'
        ? -messageText.scrollWidth
        : messageText.scrollWidth;
  };
  const updateMessagePreview = (value = text.value) => {
    const empty = !value.trim();
    messageText.textContent = empty ? 'Your message will appear here.' : value;
    messageText.classList.toggle('is-empty', empty);
    scrollMessageToEnd();
  };
  let selected = location.hash === '#text-workspace' ? 'text' : 'camera';
  function render() {
    const focused = document.activeElement;
    if (compact.matches) {
      const editing = panels.findIndex((panel) => panel.contains(focused));
      if (editing >= 0) selected = buttons[editing].dataset.workspaceView;
    }
    tabs.hidden = !compact.matches;
    textCameraControls.hidden = !compact.matches || selected !== 'text';
    messagePreview.hidden = !compact.matches || selected !== 'camera';
    updateMessagePreview();
    if (compact.matches) tabs.setAttribute('role', 'tablist');
    else tabs.removeAttribute('role');
    buttons.forEach((button, index) => {
      const active = button.dataset.workspaceView === selected,
        panel = panels[index];
      panel.hidden = compact.matches && !active;
      if (compact.matches) {
        button.setAttribute('role', 'tab');
        button.setAttribute('aria-selected', String(active));
        button.tabIndex = active ? 0 : -1;
        panel.setAttribute('role', 'tabpanel');
        panel.setAttribute('aria-labelledby', button.id);
      } else {
        button.removeAttribute('role');
        button.removeAttribute('aria-selected');
        button.removeAttribute('tabindex');
        panel.removeAttribute('role');
        if (labels[index]) panel.setAttribute('aria-labelledby', labels[index]);
        else panel.removeAttribute('aria-labelledby');
      }
    });
    if (compact.matches && panels.some((panel) => panel.hidden && panel.contains(focused)))
      buttons.find((button) => button.dataset.workspaceView === selected).focus();
    if (!compact.matches && buttons.includes(focused)) {
      const panel =
        panels[buttons.findIndex((button) => button.dataset.workspaceView === selected)];
      const target = [
        ...panel.querySelectorAll('button:not([disabled]),textarea,a[href],input'),
      ].find((element) => element.getClientRects().length);
      target?.focus({ preventScroll: true });
    }
    if (textCameraControls.hidden && textCameraControls.contains(focused)) {
      const target = compact.matches
        ? buttons.find((button) => button.dataset.workspaceView === selected)
        : cameraStop.disabled
          ? document.getElementById('translation-text')
          : cameraStop;
      target.focus({ preventScroll: true });
    }
    if (messagePreview.hidden && messagePreview.contains(focused))
      document.getElementById('text-heading').focus({ preventScroll: true });
  }
  buttons.forEach((button, index) => {
    button.addEventListener('click', () => {
      selected = button.dataset.workspaceView;
      render();
    });
    button.addEventListener('keydown', (event) => {
      let next;
      if (event.key === 'ArrowRight') next = (index + 1) % buttons.length;
      else if (event.key === 'ArrowLeft') next = (index + buttons.length - 1) % buttons.length;
      else if (event.key === 'Home') next = 0;
      else if (event.key === 'End') next = buttons.length - 1;
      else return;
      event.preventDefault();
      selected = buttons[next].dataset.workspaceView;
      render();
      buttons[next].focus();
    });
  });
  messageEdit.addEventListener('click', () => {
    selected = 'text';
    render();
    text.focus({ preventScroll: true });
    text.scrollIntoView({ block: 'nearest', inline: 'nearest' });
  });
  compact.addEventListener('change', render);
  window.addEventListener('resize', scrollMessageToEnd);
  render();
  const viewport = document.getElementById('camera-viewport');
  const indicator = document.getElementById('workspace-camera-indicator');
  const updateCamera = () => {
    indicator.hidden = !viewport.classList.contains('running');
    if (textCameraStatus.innerHTML !== cameraStatus.innerHTML) {
      textCameraStatus.replaceChildren(
        ...[...cameraStatus.childNodes].map((node) => node.cloneNode(true)),
      );
    }
    textCameraStop.disabled = cameraStop.disabled;
  };
  const cameraObserver = new MutationObserver(updateCamera);
  cameraObserver.observe(viewport, {
    attributes: true,
    attributeFilter: ['class'],
  });
  cameraObserver.observe(cameraStatus, { childList: true, subtree: true, characterData: true });
  cameraObserver.observe(cameraStop, { attributes: true, attributeFilter: ['disabled'] });
  textCameraStop.addEventListener('click', () => cameraStop.click());
  updateCamera();
  const help = document.getElementById('translator-help');
  document.getElementById('translator-help-open').addEventListener('click', () => help.showModal());
  document.getElementById('translator-help-close').addEventListener('click', () => help.close());
  return { updateMessagePreview };
}
