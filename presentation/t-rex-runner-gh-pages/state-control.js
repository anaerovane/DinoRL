(() => {
  'use strict';
  function runner() { return window.Runner && window.Runner.instance_; }
  function getState() {
    const r = runner();
    if (!r || !r.tRex || !r.horizon) return {ready:false};
    const t = r.tRex;
    return {ready:true, playing:r.playing, crashed:r.crashed, paused:r.paused,
      speed:r.currentSpeed, distance:r.distanceRan,
      score:r.distanceMeter.getActualDistance(Math.ceil(r.distanceRan)),
      dinosaur:{x:t.xPos,y:t.yPos,jumping:t.jumping,ducking:t.ducking,verticalVelocity:t.jumpVelocity},
      obstacles:r.horizon.obstacles.map(o=>({type:o.typeConfig.type,x:o.xPos,y:o.yPos,
        width:o.width,height:o.typeConfig.height,size:o.size,distance:o.xPos-(t.xPos+t.config.WIDTH)}))};
  }
  const history=[];
  function action(name) {
    const r=runner();
    if (!r || !r.tRex) return;
    if (name==='start') {
      if(r.crashed) r.restart();
      else r.onKeyDown({keyCode:32,target:document.body,preventDefault(){}});
    } else if(name==='jump') {
      r.tRex.setDuck(false); r.tRex.startJump(r.currentSpeed);
    } else if(name==='duck') r.tRex.setDuck(true);
    else if(name==='release') {r.tRex.setDuck(false);r.tRex.endJump();}
    else if(name==='pause') r.stop();
    else if(name==='resume') r.play();
    else throw new Error('Unknown action: '+name);
    const result={action:name,state:getState()};
    history.push(result);if(history.length>12)history.shift();
    render();return result;
  }
  window.dinoControl={getState,action};
  let panel,state,log;
  function render(){if(state){state.textContent=JSON.stringify(getState(),null,2);log.textContent=JSON.stringify(history,null,2);}}
  window.addEventListener('DOMContentLoaded',()=>{
    panel=document.createElement('section');
    panel.style.cssText='position:relative;margin:180px 12px 20px;background:#f2f5f9;color:#172334;padding:12px;font:13px monospace;z-index:10';
    const heading=document.createElement('h2');heading.textContent='Live game state and controls';panel.appendChild(heading);
    for(const [name,label] of [['start','Start / Restart'],['jump','Jump'],['duck','Duck'],['release','Release'],['pause','Pause'],['resume','Resume']]){
      const b=document.createElement('button');b.textContent=label;b.style.cssText='margin:4px;padding:8px';b.onclick=()=>action(name);panel.appendChild(b);
    }
    state=document.createElement('pre');state.id='dino-state';state.style.cssText='white-space:pre-wrap';panel.appendChild(state);
    const title=document.createElement('h3');title.textContent='Action snapshots';panel.appendChild(title);
    log=document.createElement('pre');log.id='dino-actions';log.style.cssText='white-space:pre-wrap';panel.appendChild(log);
    document.body.appendChild(panel);setInterval(render,100);render();
  });
})();
