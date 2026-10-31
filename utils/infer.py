"""Rule-based tech-stack inference.

infer_from_text(text) -> {
    frontend, backend, database,      # best guess or None
    confidence,                       # 0..1, mean confidence of detected layers
    evidence,                         # human-readable matched signals
    skeleton,                         # suggested project layout lines
    candidates, extras                # runners-up and UI/tooling hints
}
"""
import math
import re
from typing import Dict, List, Optional, Tuple

# (regex, weight)  weight: 3 = strong signal, 2 = medium, 1 = weak
Rule = Tuple[str, int]

FRONTEND: Dict[str, List[Rule]] = {
    "Next.js": [(r"/_next/", 3), (r"__next_data__", 3), (r"\bnext/(?:link|router|image)", 3), (r"\bnext\.js\b", 3), (r"\bnextjs\b", 2)],
    "Nuxt": [(r"/_nuxt/", 3), (r"__nuxt", 3), (r"\bnuxt(?:\.js)?\b", 2)],
    "React": [(r"react-dom", 3), (r"data-reactroot", 3), (r"__react", 2), (r"\buse(?:State|Effect|Ref|Memo|Callback)\b", 2),
              (r"from ['\"]react['\"]", 3), (r"\breact(?:\.js)?\b", 1), (r"\bjsx\b", 1)],
    "Vue": [(r"data-v-[0-9a-f]{6,}", 3), (r"__vue__", 3), (r"\bv-(?:if|for|model|bind|on)\b", 2), (r"@vue/", 3),
            (r"createApp\(", 1), (r"\bvue(?:\.js)?\b", 1), (r"\.vue\b", 2)],
    "Angular": [(r"ng-version", 3), (r"@angular/", 3), (r"\bng-(?:app|controller|model|repeat)\b", 2), (r"\*ng(?:If|For)\b", 3), (r"\bangular\b", 1)],
    "Svelte": [(r"/_app/immutable/", 3), (r"\bsveltekit\b", 3), (r"\bsvelte\b", 2), (r"\.svelte\b", 2)],
    "jQuery": [(r"jquery[\w.\-]*\.js", 3), (r"\$\(document\)\.ready", 3), (r"\bjquery\b", 2)],
    "HTMX": [(r"\bhx-(?:get|post|swap|target|trigger)\b", 3), (r"htmx(?:\.min)?\.js", 3)],
    "Alpine.js": [(r"\bx-data\b", 3), (r"alpine(?:\.min)?\.js", 3)],
}

BACKEND: Dict[str, List[Rule]] = {
    "FastAPI": [(r"\bfastapi\b", 3), (r"\buvicorn\b", 2), (r"/openapi\.json", 2), (r"swagger ui", 1), (r"\bpydantic\b", 1), (r"@app\.(?:get|post|put|delete)\(", 1)],
    "Flask": [(r"\bflask\b", 3), (r"\bwerkzeug\b", 3), (r"\bjinja2?\b", 2), (r"\bgunicorn\b", 1), (r"@app\.route\(", 2)],
    "Django": [(r"\bdjango\b", 3), (r"csrfmiddlewaretoken", 3), (r"\bmanage\.py\b", 2), (r"\bsettings\.py\b", 2), (r"\bdrf\b|rest_framework", 2)],
    "Express": [(r"x-powered-by:\s*express", 3), (r"\bexpress(?:\.js)?\b", 2), (r"require\(['\"]express['\"]\)", 3), (r"app\.listen\(", 1)],
    "NestJS": [(r"@nestjs/", 3), (r"\bnestjs\b", 3), (r"@Controller\(", 2), (r"@Injectable\(", 2)],
    "Koa": [(r"\bkoa\b", 3), (r"require\(['\"]koa['\"]\)", 3)],
    "Spring Boot": [(r"springframework", 3), (r"\bspring boot\b", 3), (r"whitelabel error page", 3), (r"jsessionid", 2),
                    (r"@(?:RestController|SpringBootApplication|Autowired)\b", 3), (r"\bspring\b", 1)],
    "Laravel": [(r"\blaravel\b", 3), (r"laravel_session", 3), (r"\billuminate\\", 3), (r"\bartisan\b", 2), (r"xsrf-token", 1)],
    "PHP": [(r"x-powered-by:\s*php", 3), (r"phpsessid", 3), (r"\.php\b", 2), (r"<\?php", 3)],
    "Ruby on Rails": [(r"ruby on rails", 3), (r"\brails\b", 2), (r"csrf-param", 3), (r"action_?controller", 3), (r"\bactiverecord\b", 2)],
    "ASP.NET": [(r"asp\.net", 3), (r"__viewstate", 3), (r"\.aspx?\b", 2), (r"x-aspnet-version", 3), (r"\.net core|aspnetcore", 3)],
    "Gin (Go)": [(r"gin-gonic", 3), (r"\bgin\.(?:default|new|context)\b", 3), (r"\bgolang\b", 1), (r"\bgorm\b", 1)],
    "Go (net/http)": [(r"net/http", 2), (r"\bhttp\.handlefunc\b", 3), (r"package main", 1)],
}

DATABASE: Dict[str, List[Rule]] = {
    "PostgreSQL": [(r"\bpostgres(?:ql)?\b", 3), (r"psycopg2?|asyncpg|node-postgres|\bpg_\w+", 3), (r"postgres://|postgresql://", 3), (r"\bjsonb\b", 2), (r"\bserial\b|\bbigserial\b", 1)],
    "MySQL": [(r"\bmysql\b", 3), (r"pymysql|mysqli|mysqlclient|mysql2", 3), (r"\bmariadb\b", 2), (r"mysql://", 3), (r"auto_increment", 1)],
    "SQLite": [(r"\bsqlite3?\b", 3), (r"\.sqlite3?\b|\.db\b", 1)],
    "MongoDB": [(r"\bmongo(?:db)?\b", 3), (r"mongoose|pymongo|motor\.motor_asyncio", 3), (r"mongodb(?:\+srv)?://", 3), (r"\bobjectid\b", 2)],
    "Redis": [(r"\bredis\b", 3), (r"redis://", 3), (r"ioredis|aioredis", 3)],
    "SQL Server": [(r"sql server|\bmssql\b|pyodbc|tedious", 3), (r"sqlserver://", 3)],
    "Oracle": [(r"\boracle\b|cx_oracle|oracledb|ora-\d{5}", 3)],
    "Elasticsearch": [(r"elasticsearch|\belastic\b|opensearch", 3)],
    "Firebase": [(r"firebase|firestore", 3)],
    "Supabase": [(r"supabase", 3)],
    "DynamoDB": [(r"dynamodb|boto3\.resource\(['\"]dynamodb", 3)],
}

EXTRAS: Dict[str, str] = {
    "Tailwind CSS": r"tailwind|\b(?:flex|grid) (?:items|justify)-|\bbg-\w+-\d{3}\b",
    "Bootstrap": r"bootstrap(?:\.min)?\.(?:css|js)|\bclass=\"[^\"]*\bcol-(?:md|lg|sm)-\d",
    "Vite": r"/@vite/|\bvite\b",
    "Webpack": r"webpack",
    "TypeScript": r"\.tsx?\b|\btypescript\b",
    "Nginx": r"\bnginx\b",
    "Cloudflare": r"cloudflare|cf-ray",
    "Docker": r"dockerfile|docker-compose|\bdocker\b",
}

SKELETONS: Dict[str, List[str]] = {
    "FastAPI": ["app.py / main.py  (FastAPI entry)", "routers/  (API routes)", "schemas.py  (pydantic models)", "database.py  (DB access)", "requirements.txt"],
    "Flask": ["app.py  (Flask entry)", "templates/  (Jinja2)", "static/", "models.py", "requirements.txt"],
    "Django": ["manage.py", "<project>/settings.py, urls.py", "<app>/models.py, views.py", "templates/", "requirements.txt"],
    "Express": ["server.js / index.js", "routes/", "controllers/", "models/", "package.json"],
    "NestJS": ["src/main.ts", "src/app.module.ts", "src/<feature>/*.controller.ts, *.service.ts", "package.json"],
    "Koa": ["app.js", "routes/", "middleware/", "package.json"],
    "Spring Boot": ["src/main/java/.../Application.java", "controller/ service/ repository/", "src/main/resources/application.yml", "pom.xml"],
    "Laravel": ["routes/web.php, api.php", "app/Http/Controllers/", "app/Models/", "database/migrations/", "composer.json"],
    "PHP": ["index.php", "includes/", "config.php", "composer.json"],
    "Ruby on Rails": ["config/routes.rb", "app/controllers/", "app/models/", "db/migrate/", "Gemfile"],
    "ASP.NET": ["Program.cs", "Controllers/", "Models/", "appsettings.json", "*.csproj"],
    "Gin (Go)": ["main.go", "handler/", "model/", "go.mod"],
    "Go (net/http)": ["main.go", "handlers/", "go.mod"],
}
FRONTEND_SKELETONS: Dict[str, List[str]] = {
    "React": ["src/index.jsx", "src/App.jsx", "src/components/", "package.json"],
    "Next.js": ["app/ or pages/", "components/", "public/", "next.config.js"],
    "Vue": ["src/main.js", "src/App.vue", "src/components/", "package.json"],
    "Nuxt": ["pages/", "components/", "nuxt.config.ts"],
    "Angular": ["src/main.ts", "src/app/app.module.ts", "angular.json"],
    "Svelte": ["src/routes/", "src/lib/", "svelte.config.js"],
}

_COMPILED_CACHE: Dict[str, List[Tuple[re.Pattern, int, str]]] = {}


def _compiled(group: str, table: Dict[str, List[Rule]], name: str):
    key = f"{group}:{name}"
    if key not in _COMPILED_CACHE:
        _COMPILED_CACHE[key] = [(re.compile(p, re.I), w, p) for p, w in table[name]]
    return _COMPILED_CACHE[key]


def _score_group(group: str, table: Dict[str, List[Rule]], text: str):
    """Return [(name, score, [evidence...])] sorted by score desc."""
    results = []
    for name in table:
        score, hits = 0, []
        for rx, weight, raw in _compiled(group, table, name):
            m = rx.search(text)
            if m:
                score += weight
                snippet = m.group(0).strip()[:40]
                hits.append(f"{name}: 命中「{snippet}」(权重 {weight})")
        if score:
            results.append((name, score, hits))
    results.sort(key=lambda r: r[1], reverse=True)
    return results


def _to_confidence(score: int, runner_up: int = 0) -> float:
    """Saturating score, penalised when a competitor is close."""
    base = 1 - math.exp(-score / 4.0)
    if runner_up:
        base *= max(0.5, 1 - runner_up / (score + runner_up + 1) * 0.8)
    return round(min(base, 0.99), 2)


def infer_from_text(text: Optional[str]) -> dict:
    text = (text or "")[:300_000]
    layers = {
        "frontend": _score_group("fe", FRONTEND, text),
        "backend": _score_group("be", BACKEND, text),
        "database": _score_group("db", DATABASE, text),
    }

    result: dict = {"frontend": None, "backend": None, "database": None}
    evidence: List[str] = []
    confidences: List[float] = []
    candidates: Dict[str, list] = {}

    for layer, ranked in layers.items():
        if not ranked:
            candidates[layer] = []
            continue
        best_name, best_score, hits = ranked[0]
        runner = ranked[1][1] if len(ranked) > 1 else 0
        result[layer] = best_name
        confidences.append(_to_confidence(best_score, runner))
        evidence.extend(hits[:4])
        candidates[layer] = [{"name": n, "score": s} for n, s, _ in ranked[:3]]

    extras = [name for name, pat in EXTRAS.items() if re.search(pat, text, re.I)]

    skeleton: List[str] = []
    if result["frontend"] in FRONTEND_SKELETONS:
        skeleton.append(f"# 前端 ({result['frontend']})")
        skeleton.extend(FRONTEND_SKELETONS[result["frontend"]])
    if result["backend"] in SKELETONS:
        skeleton.append(f"# 后端 ({result['backend']})")
        skeleton.extend(SKELETONS[result["backend"]])
    if result["database"]:
        skeleton.append(f"# 数据库 ({result['database']})")
        skeleton.append("schema / migrations 与连接配置（环境变量 DATABASE_URL）")

    if not evidence:
        evidence.append("未发现明确的技术栈特征，请提供更多代码、页面源码或截图。")

    result.update(
        confidence=round(sum(confidences) / len(confidences), 2) if confidences else 0.0,
        evidence=evidence,
        skeleton=skeleton,
        candidates=candidates,
        extras=extras,
    )
    return result
