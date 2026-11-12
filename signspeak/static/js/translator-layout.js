// On narrow screens, switch workspaces without scrolling the document.
export function initTranslatorLayout() {
  const tabs=document.querySelector('.workspace-tabs');
  const buttons=[...tabs.querySelectorAll('[data-workspace-view]')];
  const panels=buttons.map(button=>document.getElementById(button.getAttribute('aria-controls')));
  const labels=panels.map(panel=>panel.getAttribute('aria-labelledby'));
  const compact=matchMedia('(max-width: 980px)');
  let selected=location.hash==='#text-workspace'?'text':'camera';
  function render() {
    const focused=document.activeElement;
    if(compact.matches){
      const editing=panels.findIndex(panel=>panel.contains(focused));
      if(editing>=0)selected=buttons[editing].dataset.workspaceView;
    }
    tabs.hidden=!compact.matches;
    if(compact.matches)tabs.setAttribute('role','tablist');else tabs.removeAttribute('role');
    buttons.forEach((button,index)=>{
      const active=button.dataset.workspaceView===selected,panel=panels[index];
      panel.hidden=compact.matches&&!active;
      if(compact.matches){
        button.setAttribute('role','tab');button.setAttribute('aria-selected',String(active));button.tabIndex=active?0:-1;
        panel.setAttribute('role','tabpanel');panel.setAttribute('aria-labelledby',button.id);
      }else{
        button.removeAttribute('role');button.removeAttribute('aria-selected');button.removeAttribute('tabindex');
        panel.removeAttribute('role');
        if(labels[index])panel.setAttribute('aria-labelledby',labels[index]);else panel.removeAttribute('aria-labelledby');
      }
    });
    if(compact.matches&&panels.some(panel=>panel.hidden&&panel.contains(focused)))buttons.find(button=>button.dataset.workspaceView===selected).focus();
    if(!compact.matches&&buttons.includes(focused)){
      const panel=panels[buttons.findIndex(button=>button.dataset.workspaceView===selected)];
      const target=[...panel.querySelectorAll('button:not([disabled]),textarea,a[href],input')].find(element=>element.getClientRects().length);
      target?.focus({preventScroll:true});
    }
  }
  buttons.forEach((button,index)=>{
    button.addEventListener('click',()=>{selected=button.dataset.workspaceView;render();});
    button.addEventListener('keydown',event=>{
      let next;
      if(event.key==='ArrowRight')next=(index+1)%buttons.length;
      else if(event.key==='ArrowLeft')next=(index+buttons.length-1)%buttons.length;
      else if(event.key==='Home')next=0;
      else if(event.key==='End')next=buttons.length-1;
      else return;
      event.preventDefault();selected=buttons[next].dataset.workspaceView;render();buttons[next].focus();
    });
  });
  compact.addEventListener('change',render);render();
  const viewport=document.getElementById('camera-viewport');
  const indicator=document.getElementById('workspace-camera-indicator');
  const updateCamera=()=>{indicator.hidden=!viewport.classList.contains('running');};
  new MutationObserver(updateCamera).observe(viewport,{attributes:true,attributeFilter:['class']});updateCamera();
  const help=document.getElementById('translator-help');
  document.getElementById('translator-help-open').addEventListener('click',()=>help.showModal());
  document.getElementById('translator-help-close').addEventListener('click',()=>help.close());
}
