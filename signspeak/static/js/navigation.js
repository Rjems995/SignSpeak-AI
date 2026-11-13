// Keep the mobile drawer out of the tab order until it is opened. Desktop
// navigation remains available, including after a viewport change.
export function initNavigation() {
  const sidebar = document.getElementById('sidebar');
  const toggle = document.getElementById('menu-toggle');
  const close = document.getElementById('menu-close');
  const backdrop = document.getElementById('nav-backdrop');
  const shell = document.querySelector('.app-shell');
  if (!sidebar || !toggle || !close || !backdrop || !shell) return;
  const mobile = window.matchMedia('(max-width: 760px)');
  let opened = false;
  const focusable = () =>
    [
      ...sidebar.querySelectorAll(
        'a[href],button:not([disabled]),input:not([disabled]),select:not([disabled]),textarea:not([disabled]),[tabindex]:not([tabindex="-1"])',
      ),
    ].filter((element) => element.getClientRects().length && !element.closest('[inert]'));

  function render() {
    const modal = mobile.matches && opened;
    sidebar.classList.toggle('open', modal);
    sidebar.inert = mobile.matches && !modal;
    if (mobile.matches) sidebar.setAttribute('aria-hidden', String(!modal));
    else sidebar.removeAttribute('aria-hidden');
    if (modal) {
      sidebar.setAttribute('role', 'dialog');
      sidebar.setAttribute('aria-modal', 'true');
    } else {
      sidebar.removeAttribute('role');
      sidebar.removeAttribute('aria-modal');
    }
    shell.inert = modal;
    backdrop.hidden = !modal;
    document.body.classList.toggle('nav-open', modal);
    toggle.setAttribute('aria-expanded', String(modal));
    toggle.setAttribute('aria-label', modal ? 'Close navigation' : 'Open navigation');
  }

  function setOpen(next, { restoreFocus = true } = {}) {
    const wasOpen = opened;
    opened = mobile.matches && next;
    // Move focus before hiding its ancestor from assistive technology.
    if (wasOpen && !opened) {
      shell.inert = false;
      if (restoreFocus && mobile.matches) toggle.focus();
      else if (sidebar.contains(document.activeElement)) document.activeElement.blur();
    }
    render();
    if (opened) close.focus();
  }

  toggle.addEventListener('click', () => setOpen(!opened));
  close.addEventListener('click', () => setOpen(false));
  backdrop.addEventListener('click', () => setOpen(false));
  sidebar.addEventListener('click', (event) => {
    if (event.target.closest('a[href]') && opened) setOpen(false, { restoreFocus: false });
  });
  document.addEventListener('keydown', (event) => {
    if (!opened) return;
    if (event.key === 'Escape') {
      event.preventDefault();
      setOpen(false);
    } else if (event.key === 'Tab') {
      const items = focusable(),
        first = items[0],
        last = items.at(-1);
      if (!first) {
        event.preventDefault();
        return;
      }
      if (!sidebar.contains(document.activeElement)) {
        event.preventDefault();
        (event.shiftKey ? last : first).focus();
      } else if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }
  });
  mobile.addEventListener('change', () => {
    const sidebarHadFocus = sidebar.contains(document.activeElement);
    const closeHadFocus = document.activeElement === close;
    opened = false;
    shell.inert = false;
    if (mobile.matches && sidebarHadFocus) toggle.focus();
    render();
    if (!mobile.matches && closeHadFocus) {
      (sidebar.querySelector('[aria-current="page"]') || sidebar.querySelector('a[href]')).focus();
    }
  });
  render();
}
