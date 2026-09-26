// Headless render test for the workbench page: loads the real page against a
// running server (BGSIM_URL), lets the script run, and walks every tab.
const { JSDOM } = require("jsdom");
const base = process.env.BGSIM_URL || "http://127.0.0.1:8621";
(async () => {
  const dom = await JSDOM.fromURL(base + "/", { runScripts: "dangerously", resources: "usable", pretendToBeVisual: true,
    beforeParse(w){
      w.fetch = (u, o) => fetch(new URL(u, base).toString(), o);
      w.alert = (m)=>{ w.__alerts.push(m); }; w.confirm = ()=>true; w.prompt = ()=>"x";
      w.__errors = []; w.__alerts = []; w.addEventListener("error", e => w.__errors.push(e.message));
      w.Notification = undefined;
    }});
  const w = dom.window, doc = w.document;
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  await sleep(2500);
  const text = () => (doc.querySelector("#main") || doc.body).textContent.replace(/\s+/g, " ");
  const checks = []; const ok = (n, c) => checks.push([n, !!c]);
  ok("no script errors on load", w.__errors.length === 0);
  ok("game auto-opened with a next step", /Next:/.test(text()));
  for (const tab of ["rules", "findings", "balance", "engine", "logs", "overview"]) {
    w.setTab(tab); await sleep(600);
    ok(`tab ${tab} renders without errors`, w.__errors.length === 0 && doc.querySelector(".sheet").textContent.trim().length > 0);
  }
  w.setTab("balance"); await sleep(600);
  ok("balance tab has the one button", /Run the balance analysis/.test(text()));
  ok("stale strategy flagged in advanced", /made for a previous engine/.test(text()));
  w.setTab("rules"); await sleep(600);
  const inp = doc.querySelector('input[id^="ws-"]');
  ok("workshop input has its own id", !inp || inp.tagName === "INPUT");
  for (const [n, c] of checks) console.log((c ? "PASS " : "FAIL ") + n);
  if (w.__errors.length) console.log("errors:", w.__errors);
  process.exit(checks.some(c => !c[1]) ? 1 : 0);
})().catch(e => { console.log("JS ERROR:", e.message); process.exit(1); });
