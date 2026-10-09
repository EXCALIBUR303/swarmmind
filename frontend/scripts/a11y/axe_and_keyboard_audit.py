import asyncio, json, urllib.request, sys
import websockets
PAGES=[sys.argv[3]]
W=int(sys.argv[1]); MODE=sys.argv[2]
AXE=open("/Users/sid/Claude/swarmmind/frontend/node_modules/axe-core/axe.min.js").read()
async def main():
    tabs=json.load(urllib.request.urlopen("http://127.0.0.1:9333/json"))
    page=[t for t in tabs if t["type"]=="page"][0]
    async with websockets.connect(page["webSocketDebuggerUrl"],max_size=None) as ws:
        n=0
        async def cmd(method, **params):
            nonlocal n; n+=1; i=n
            await ws.send(json.dumps({"id":i,"method":method,"params":params}))
            while True:
                m=json.loads(await ws.recv())
                if m.get("id")==i: return m.get("result")
        async def js(expr):
            r=await cmd("Runtime.evaluate",expression=expr,awaitPromise=True,returnByValue=True); return r.get("result",{}).get("value")
        await cmd("Emulation.setDeviceMetricsOverride",width=W,height=900,deviceScaleFactor=1,mobile=False)
        for path in PAGES:
            await cmd("Page.navigate",url="http://localhost:5173"+path)
            await asyncio.sleep(3.5); print("loaded",path,flush=True)
            if MODE=="axe":
                await js(AXE)
                res=await js("axe.run(document,{runOnly:{type:'tag',values:['wcag2a','wcag2aa','wcag21a','wcag21aa','wcag22aa','best-practice']}}).then(r=>r.violations.map(v=>({id:v.id,impact:v.impact,n:v.nodes.length,ex:v.nodes.slice(0,2).map(x=>x.target.join(' ')+' :: '+(x.any[0]?.message||x.all[0]?.message||'').slice(0,160))})))")
                print(W,path,"axe violations:",len(res or []))
                for v in res or []: print("   ",v)
            else:
                total=await js("[...document.querySelectorAll('a[href],button,input,select,textarea,summary,[tabindex=\"0\"]')].filter(e=>!e.disabled&&e.offsetParent!==null&&e.tabIndex>=0&&getComputedStyle(e).visibility!=='hidden').length")
                seen=[];bad=[]
                for i in range(total+3):
                    await cmd("Input.dispatchKeyEvent",type="keyDown",key="Tab",code="Tab",windowsVirtualKeyCode=9)
                    await cmd("Input.dispatchKeyEvent",type="keyUp",key="Tab",code="Tab",windowsVirtualKeyCode=9)
                    info=await js("(()=>{const e=document.activeElement;if(!e||e===document.body)return null;const again=e.dataset.v==='1';e.dataset.v='1';const s=getComputedStyle(e);const r=e.getBoundingClientRect();return {again,tag:e.tagName,name:(e.getAttribute('aria-label')||e.innerText||e.id||'').slice(0,30),outline:s.outlineStyle+' '+s.outlineWidth,shadow:s.boxShadow!=='none',w:Math.round(r.width),h:Math.round(r.height)}})()")
                    if info is None: continue
                    if info["again"]: break
                    seen.append(info["name"])
                    if info["outline"].startswith("none") and not info["shadow"]: bad.append(info)
                print(W,path,"interactive:",total,"tab stops visited:",len(seen),"NO focus indicator:",len(bad),bad[:3])
asyncio.run(main())
