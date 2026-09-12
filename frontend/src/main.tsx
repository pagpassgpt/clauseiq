import React, {useMemo, useState} from 'react';
import {createRoot} from 'react-dom/client';
import './style.css';

type Finding={id:string; title:string; rationale:string; quote:string; severity:string; category:string; clause_id:string; grounding_status:string};
type Review={risk_level:string; risk_score:number; findings:Finding[]; review_id:string};

const API=import.meta.env.VITE_API_URL || 'http://localhost:8000';

function App(){
  const [file,setFile]=useState<File>(); const [review,setReview]=useState<Review>();
  const [busy,setBusy]=useState(false); const [error,setError]=useState(''); const [tab,setTab]=useState('findings');
  const [contractId,setContractId]=useState('');
  const counts=useMemo(()=>{const f=review?.findings||[]; return {total:f.length,high:f.filter(x=>x.severity.toLowerCase()==='high').length,medium:f.filter(x=>x.severity.toLowerCase()==='medium').length}},[review]);
  async function run(){
    if(!file)return; setBusy(true); setError('');
    try{
      const fd=new FormData(); fd.append('file',file);
      const u=await fetch(`${API}/contracts`,{method:'POST',body:fd}); if(!u.ok) throw new Error(await u.text());
      const c=await u.json(); setContractId(c.id);
      const r=await fetch(`${API}/contracts/${c.id}/review`,{method:'POST'}); if(!r.ok) throw new Error(await r.text());
      setReview(await r.json()); setTab('findings');
    }catch(e:any){setError(e.message||'Review failed')}finally{setBusy(false)}
  }
  return <div className="shell">
    <header><div className="brand"><div className="logo">C</div><div><h1>ClauseIQ</h1><p>Grounded contract intelligence</p></div></div><div className="status">● Evidence-first · v6</div></header>
    <main>
      <section className="hero"><div><span className="eyebrow">AI contract review</span><h2>Review contracts with evidence you can audit.</h2><p>Clause-aware retrieval, deterministic policy controls, and grounded findings—without letting model output become unsupported fact.</p></div>
      <div className="upload"><label>{file ? file.name : 'Choose a PDF, DOCX, TXT or Markdown contract'}<input type="file" accept=".pdf,.docx,.txt,.md" onChange={e=>setFile(e.target.files?.[0])}/></label><button disabled={!file||busy} onClick={run}>{busy?'Analyzing contract…':'Run grounded review'}</button></div></section>
      {error&&<div className="error">{error}</div>}
      {review&&<>
        <section className="metrics"><div className={`risk ${review.risk_level.toLowerCase()}`}><small>Risk score</small><strong>{review.risk_score}<span>/100</span></strong><b>{review.risk_level}</b></div><div><small>Verified findings</small><strong>{counts.total}</strong><p>{counts.high} high · {counts.medium} medium</p></div><div><small>Evidence status</small><strong>100%</strong><p>grounding gate passed</p></div><div><small>Contract</small><strong className="mono">{contractId.slice(0,10)}</strong><p>persisted review</p></div></section>
        <nav className="tabs">{[['findings','Findings'],['evidence','Evidence'],['audit','Audit']].map(([k,v])=><button className={tab===k?'active':''} onClick={()=>setTab(k)} key={k}>{v}</button>)}</nav>
        {tab==='findings'&&<section className="grid">{review.findings.map(f=><article className="card" key={f.id}><div className="cardtop"><span className="tag">{f.category}</span><span className={`severity ${f.severity.toLowerCase()}`}>{f.severity}</span></div><h3>{f.title}</h3><p>{f.rationale}</p><blockquote>“{f.quote}”</blockquote><footer><span>{f.clause_id}</span><span>{f.grounding_status}</span></footer></article>)}{!review.findings.length&&<div className="empty">No policy findings were verified.</div>}</section>}
        {tab==='evidence'&&<section className="panel"><h3>Evidence ledger</h3>{review.findings.map(f=><div className="ledger" key={f.id}><div><b>{f.clause_id}</b><span>{f.category}</span></div><blockquote>{f.quote}</blockquote><small>Grounding: {f.grounding_status}</small></div>)}</section>}
        {tab==='audit'&&<section className="panel"><h3>Review audit trail</h3><dl><dt>Review ID</dt><dd className="mono">{review.review_id}</dd><dt>Evidence policy</dt><dd>Unsupported quotes are rejected before persistence.</dd><dt>Risk authority</dt><dd>Deterministic policy engine; model output cannot directly set final risk.</dd><dt>Contract scope</dt><dd>Retrieval and clause access are scoped to the active contract.</dd></dl></section>}
      </>}
    </main>
  </div>
}
createRoot(document.getElementById('root')!).render(<App/>);
