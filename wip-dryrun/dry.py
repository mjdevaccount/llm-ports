"""DEV round dry-run: the claim judge over the replayable gold, on the deployed box model. Read-only (no rows)."""
import asyncio, datetime as dt, json, re, sys, collections
import llm_ports.evals as E
E.__path__.append("/tmp/cj")
from llm_ports.evals import claim_judge as cj, decompose as d
from llm_ports.evals.judge import schema_of
from llm_ports.learn.examples import Agreement
from service.platform import llm_call, llm_roles
from service.platform import llm_judge_inference_rules as inf
from service.evals import judge_seen, judge_gold

TEMPLATE = open("/tmp/cj/prompt.j2", encoding="utf-8").read()
FIELDS = json.load(open("/tmp/cj/fields.json", encoding="utf-8"))
import jinja2
TPL = jinja2.Environment(undefined=jinja2.StrictUndefined, keep_trailing_newline=True).from_string(TEMPLATE)

class P:
    def render(self, name, **ctx): return TPL.render(**ctx)

def spec(role, leaf):
    m = FIELDS.get(role, {})
    f = leaf.field if not leaf.path.startswith("sentence[") else "sentence"
    return m.get(f) or m.get("*") or {}

async def one(case, judge, static):
    role = case["source"]["role"]
    system, output = static[role]() if role in static else ("", None)
    schema = schema_of(output)
    allowed = d.enums(schema)
    rules = inf.rules_for(role)
    ctx = (f"Today is {case['today']}. The reply is from the `{role}` role.\n"
           + (("ROLE INSTRUCTIONS (given to the model):\n" + system[:2500] + "\n") if system else "")
           + (("PLATFORM RULES:\n" + "\n".join(f"- {r}" for r in rules) + "\n") if rules else ""))
    def kind_of(leaf):
        k = spec(role, leaf).get("kind")
        if k: return k
        return "classify" if leaf.field in allowed else ("free" if leaf.path.startswith("sentence[") else "extract")
    def material_of(leaf):
        return spec(role, leaf).get("material", True)
    def question_of(leaf, kind):
        q = spec(role, leaf).get("ask")
        return q or cj.question_for(leaf, kind)
    def pre(leaf, kind):
        if spec(role, leaf).get("names"):  # a symbol: supported when a header lists it or a cashtag names it
            v = leaf.value.strip()
            for line in case["input"].splitlines():
                if re.match(r"\s*(TICKERS|CANDIDATE SYMBOLS?)\s*:", line, re.I) and re.search(rf"(?<![A-Z0-9]){re.escape(v)}(?![A-Z0-9])", line):
                    return d.Prematch("supported", "the header lists it")
            if re.search(rf"\${re.escape(v)}(?![A-Za-z0-9])", case["input"]):
                return d.Prematch("supported", "a cashtag names it")
        return None
    r = await cj.grade(case["output"], case["input"], judge=judge, kind_of=kind_of, material_of=material_of,
                       allowed=allowed, context=ctx, pre=pre, question_of=question_of)
    return r

async def main():
    cases = judge_gold.load_cases()
    static = judge_seen.static_seen()
    role = llm_roles.LLM_JUDGE
    judge = cj.GenerateClaimJudge(llm_call.generate, role, pack=P())
    out = {}
    only = sys.argv[1:]
    for cid, case in sorted(cases.items()):
        if only and not any(o in cid for o in only): continue
        r = await one(case, judge, static)
        out[cid] = r
        fails = [(v.path, v.label, v.quote[:60], v.reason[:90]) for v in r.verdicts if v.material and v.label in ("not_in_input","contradicted")]
        print(f"{cid} gold={case['case_label']} got={r.label} {r.counts()}", flush=True)
        for f in fails: print("   FAIL", f, flush=True)
    yn = lambda v: "yes" if v == "pass" else "no"
    ran = {c: r.label for c, r in out.items() if r.label in ("pass", "fail")}
    a = Agreement.of([(yn(v), yn(cases[c]["case_label"])) for c, v in ran.items()])
    gp = [c for c in ran if cases[c]["case_label"] == "pass"]; gf = [c for c in ran if cases[c]["case_label"] != "pass"]
    print("ran", len(ran), "of", len(out), "unjudgeable", [c for c, r in out.items() if r.label == "unjudgeable"])
    print("kappa", a.kappa, a.kappa_low, a.kappa_high, "agree", a.agree, "ac1", a.ac1)
    print("pass_recall", sum(ran[c] == "pass" for c in gp), "/", len(gp), "fail_catch", sum(ran[c] == "fail" for c in gf), "/", len(gf))
    tot = collections.Counter()
    for r in out.values(): tot.update(r.counts())
    print(dict(tot))

asyncio.run(main())
