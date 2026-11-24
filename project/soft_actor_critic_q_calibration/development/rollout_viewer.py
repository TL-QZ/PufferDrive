"""Bundle SAC first-action replays into the repository's standard HTML viewer."""

import base64
import json
import math
import re
import tempfile
from pathlib import Path


ACTION_COUNT = 12
DEFAULT_METRIC_LABELS = ("Q min", "training target", "discounted reward return")
JERK_LONG = (-15, -4, 0, 4)
JERK_LAT = (-4, 0, 4)
PAYLOAD_CHUNK_SIZE = 4 * 1024 * 1024
PAYLOAD_CHUNK_PATTERN = re.compile(
    r'<script type="application/octet-stream" class="payload-chunk">.*?</script>',
    re.DOTALL,
)


def write_viewer(payload, path):
    """Write one SAC standard viewer with a selector for all forced first actions.

    ``payload.actions`` contains one SAC interactive-replay binary per forced
    first action. The normal PufferLib renderer supplies the HTML and renderer;
    this module only bundles replay payloads and adds the initial-action selector.
    """
    _validate_payload(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    html = _standard_template(payload["actions"][0]["replay_bytes"])
    html = _bundle_replay_payloads(html, payload["actions"])
    metric_labels = payload.get("metric_labels", DEFAULT_METRIC_LABELS)
    scores = _display_scores(payload)
    html = _add_action_selector(html, scores, metric_labels)
    html = _add_comparison_view(html)
    endings = [action["final_outcome"] for action in payload["actions"]]
    html = _replace_once(
        html, '<span class="speed-unit">km/h</span>',
        '<span class="speed-unit">km/h</span><span id="probe-ending" class="warn-chip" hidden></span>',
        "telemetry speed",
    )
    html = _replace_once(
        html, "H.buffer = buf;",
        f"H.probeEnding = {json.dumps(endings)}[actionIndex]; H.buffer = buf;",
        "loaded branch outcome",
    )
    html = _replace_once(
        html, "            updatePolicy(f, agent);",
        "            const ending = document.getElementById('probe-ending');\n"
        "            ending.hidden = agent.id !== 0 || f !== frameMax();\n"
        "            ending.textContent = ending.hidden ? '' : H.probeEnding;\n"
        "            updatePolicy(f, agent);",
        "final frame telemetry",
    )
    destination.write_text(html, encoding="utf-8")
    return destination


def _validate_payload(payload):
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dictionary")
    actions = payload.get("actions")
    if not isinstance(actions, list) or len(actions) != ACTION_COUNT:
        raise ValueError(f"payload.actions must contain {ACTION_COUNT} forced actions")

    metric_labels = payload.get("metric_labels", DEFAULT_METRIC_LABELS)
    if (
        not isinstance(metric_labels, (list, tuple))
        or len(metric_labels) != 3
        or any(not isinstance(label, str) or not label for label in metric_labels)
        or any(not re.fullmatch(r"[A-Za-z0-9 ()/._-]+", label) for label in metric_labels)
    ):
        raise ValueError("metric_labels must contain three plain text labels")

    probabilities = payload.get("initial_probabilities")
    if not isinstance(probabilities, (list, tuple)) or len(probabilities) != ACTION_COUNT:
        raise ValueError(f"payload.initial_probabilities must contain {ACTION_COUNT} values")
    validated_probabilities = []
    for action_index, probability in enumerate(probabilities):
        if isinstance(probability, bool):
            raise ValueError(f"initial probability {action_index} must be finite and in [0, 1]")
        probability = float(probability)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"initial probability {action_index} must be finite and in [0, 1]")
        validated_probabilities.append(probability)

    display_value_presence = []
    for expected_index, action in enumerate(actions):
        if not isinstance(action, dict) or int(action.get("action_idx", -1)) != expected_index:
            raise ValueError("actions must be ordered by action_idx from 0 through 11")
        replay_bytes = action.get("replay_bytes")
        if not isinstance(replay_bytes, (bytes, bytearray, memoryview)) or not replay_bytes:
            raise ValueError(f"action {expected_index} must contain nonempty SAC replay bytes")
        action_probability = action.get("initial_probability")
        if action_probability is not None:
            action_probability = float(action_probability)
            if not math.isfinite(action_probability) or not math.isclose(
                action_probability, validated_probabilities[expected_index], rel_tol=1e-5, abs_tol=1e-7
            ):
                raise ValueError(f"action {expected_index} probability does not match initial_probabilities")

        values = action.get("display_values")
        display_value_presence.append(values is not None)
        if values is not None:
            if not isinstance(values, (list, tuple)) or len(values) != 3:
                raise ValueError(f"action {expected_index} display_values must contain three values")
            for metric_value in values:
                if metric_value is not None and not math.isfinite(float(metric_value)):
                    raise ValueError(f"action {expected_index} display_values must be finite or null")
        elif not {"q_min", "actual_critic_target", "recorded_return"}.issubset(action):
            raise ValueError(
                f"action {expected_index} must provide display_values or SAC Q/target/return fields"
            )

    if any(display_value_presence) and not all(display_value_presence):
        raise ValueError("all actions must use display_values when any action provides them")


def _display_scores(payload):
    if all("display_values" in action for action in payload["actions"]):
        return [[None if value is None else float(value) for value in action["display_values"]]
                for action in payload["actions"]]
    return [
        [action["q_min"], action["actual_critic_target"], action["recorded_return"]]
        for action in payload["actions"]
    ]



def _standard_template(first_replay_bytes):
    """Render the shared SAC page once to obtain its standard HTML template."""
    from pufferlib.sac_viz import render_interactive_replay_zlib

    with tempfile.TemporaryDirectory(prefix="qcal-render-") as temporary_directory:
        temporary_path = Path(temporary_directory)
        replay_path = temporary_path / "first-action.zlib"
        html_path = temporary_path / "standard.html"
        replay_path.write_bytes(first_replay_bytes)
        render_interactive_replay_zlib(replay_path, html_path)
        return html_path.read_text(encoding="utf-8")


def _bundle_replay_payloads(html, actions):
    existing_chunks = PAYLOAD_CHUNK_PATTERN.findall(html)
    if not existing_chunks:
        raise RuntimeError("standard SAC viewer did not contain its replay payload chunk")

    payload_chunks = []
    for action in actions:
        encoded_payload = base64.b64encode(bytes(action["replay_bytes"])).decode("ascii")
        for chunk_start in range(0, len(encoded_payload), PAYLOAD_CHUNK_SIZE):
            chunk = encoded_payload[chunk_start:chunk_start + PAYLOAD_CHUNK_SIZE]
            payload_chunks.append(
                f'<script type="application/octet-stream" class="payload-chunk" '
                f'data-action="{int(action["action_idx"])}">{chunk}</script>'
            )
    if not payload_chunks:
        raise ValueError("no replay payload chunks were generated")
    replacement = "\n".join(payload_chunks)
    first_match = PAYLOAD_CHUNK_PATTERN.search(html)
    if first_match is None:
        raise RuntimeError("standard SAC viewer replay payload could not be located")
    html = PAYLOAD_CHUNK_PATTERN.sub("", html)
    insertion_point = first_match.start()
    return html[:insertion_point] + replacement + html[insertion_point:]


def _add_action_selector(html, scores, metric_labels=DEFAULT_METRIC_LABELS):
    labels_json = json.dumps(list(metric_labels), separators=(",", ":"))
    short_label_names = {
        "V(s)": "V(s)",
        "PPO target": "Tgt",
        "discounted return": "Ret",
        "Q min": "Q",
        "training target": "Tgt",
        "discounted reward return": "Ret",
    }
    short_labels_json = json.dumps(
        [short_label_names.get(label, label) for label in metric_labels], separators=(",", ":")
    )
    html = _replace_once(
        html,
        '<div class="label">Policy</div><div id="policy-grid" class="grid"></div>',
        '<div class="label">Policy</div><div id="policy-grid" class="grid"></div>'
        f'<div class="label">Forced first action · {" / ".join(metric_labels)}</div>'
        '<div id="forced-action-grid" class="heat" aria-label="Forced first action selector"></div>'
        '<div class="heat-cap">jerk_long &#8595; / jerk_lat &#8594; · lighter = higher target</div>',
        "standard policy grid markup",
    )
    html = _replace_once(
        html,
        "        .heat-cap { grid-column:1 / -1; color:var(--muted); font-size:9.5px; font-weight:600; letter-spacing:.08em; text-transform:uppercase; }",
        "        .heat-cap { grid-column:1 / -1; color:var(--muted); font-size:9.5px; font-weight:600; letter-spacing:.08em; text-transform:uppercase; }\n"
        "        #forced-action-grid .heat-cell { width:100%; height:58px; padding:3px 6px; cursor:pointer; display:grid; grid-template-columns:auto auto; gap:0 4px; justify-content:space-evenly; font-size:11px; color:white; background:#51545a; }\n"
        "        #forced-action-grid .heat-cell:focus-visible { outline:2px solid #fff; outline-offset:1px; }",
        "standard heatmap style",
    )
    html = _replace_regex_once(
        html,
        re.compile(r"        const METRIC_LABELS = \[[^\n]*\];"),
        f"        const ACTION_SCORES = {json.dumps(scores, allow_nan=False, separators=(',', ':'))};\n"
        f"        const ACTION_METRIC_LABELS = {labels_json};\n"
        f"        const ACTION_BUTTON_LABELS = {short_labels_json};",
        "standard replay metric constants",
    )
    html = _replace_once(
        html,
        "        let step = 0, play = false, speed = 4, lastTick = 0;",
        "        let step = 0, play = false, speed = 4, lastTick = 0;\n"
        "        let selectedForcedAction = 0, replayLoadGeneration = 0, replayLoading = false;",
        "standard playback state",
    )
    html = _replace_once(
        html,
        "            const nodes = Array.from(document.querySelectorAll('.payload-chunk'));",
        "            const nodes = Array.from(document.querySelectorAll('.payload-chunk[data-action=\\\"' + actionIndex + '\\\"]'));",
        "replay payload selector",
    )
    html = _replace_once(
        html,
        "        async function decodeReplayPayload() {",
        "        async function decodeReplayPayload(actionIndex) {",
        "replay decoder signature",
    )
    html = _replace_once(
        html,
        "                    node.textContent = '';\n",
        "",
        "replay payload retention",
    )
    html = _replace_once(
        html,
        "                        for (const node of nodes) node.remove();\n",
        "",
        "replay payload retention",
    )
    html = _replace_once(
        html,
        "        async function initReplay() {\n            const buf = await decodeReplayPayload();",
        "        async function initReplay(actionIndex = 0, requestedStep = step) {\n"
        "            const generation = ++replayLoadGeneration;\n"
        "            play = false; replayLoading = true; updateBtn();\n"
        "            const buf = await decodeReplayPayload(actionIndex);\n"
        "            if (generation !== replayLoadGeneration) return;\n"
        "            replayLoading = false;",
        "replay initializer signature",
    )
    html = _replace_once(
        html,
        "            for (const name of Object.keys(H.chunks)) C[name] = chunk(name);",
        "            C = {}; for (const name of Object.keys(H.chunks)) C[name] = chunk(name);",
        "replay buffer replacement",
    )
    html = _replace_once(
        html,
        "            document.getElementById('sld').max = frameMax();\n            document.getElementById('stepTotal').textContent = frameMax();\n            updateBtn();\n            const first = getFrameAgents(0)[0]; if (first) { cam.x = first.x; cam.y = first.y; }",
        "            step = Math.max(0, Math.min(requestedStep, frameMax()));\n"
        "            document.getElementById('sld').max = frameMax();\n"
        "            document.getElementById('sld').value = step;\n"
        "            document.getElementById('stepTotal').textContent = frameMax();\n"
        "            updateBtn();\n"
        "            const first = getFrameAgents(0).find(agent => agent.slot === 0) || getFrameAgents(0)[0];\n"
        "            followedId = first ? first.id : null; isEgoCam = false;\n"
        "            if (first) { cam.x = first.x; cam.y = first.y; }\n"
        "            selectedForcedAction = actionIndex; buildForcedActionSelector();",
        "standard replay initialization",
    )
    html = _replace_once(
        html,
        "        function heatColor(t) {",
        "        function buildForcedActionSelector() {\n"
        "            const grid = document.getElementById('forced-action-grid');\n"
        "            if (!grid) return;\n"
        f"            const longValues = {json.dumps(JERK_LONG)}, latValues = {json.dumps(JERK_LAT)};\n"
        "            const targets = ACTION_SCORES.map(score => score[1]).filter(Number.isFinite);\n"
        "            const low = Math.min(...targets), high = Math.max(...targets);\n"
        "            grid.style.gridTemplateColumns = 'auto repeat(3,minmax(0,1fr))';\n"
        "            let markup = '<div class=\\\"heat-lab\\\"></div>' + latValues.map(value => '<div class=\\\"heat-lab\\\">' + value.toFixed(1) + '</div>').join('');\n"
        "            for (let row = 0; row < longValues.length; row++) {\n"
        "                markup += '<div class=\\\"heat-lab\\\">' + longValues[row].toFixed(1) + '</div>';\n"
        "                for (let column = 0; column < latValues.length; column++) {\n"
        "                    const actionIndex = row * latValues.length + column;\n"
        "                    const [value0, value1, value2] = ACTION_SCORES[actionIndex];\n"
        "                    const [label0, label1, label2] = ACTION_METRIC_LABELS;\n"
        "                    const [buttonLabel0, buttonLabel1, buttonLabel2] = ACTION_BUTTON_LABELS;\n"
        "                    const label = 'Action ' + actionIndex + ': jerk_long ' + longValues[row] + ', jerk_lat ' + latValues[column] + ' · ' + label0 + ': ' + (value0 ?? 'unavailable') + ' · ' + label1 + ': ' + (value1 ?? 'unavailable') + ' · ' + label2 + ': ' + (value2 ?? 'unavailable');\n"
        "                    const level = high > low ? (value1 - low) / (high - low) : 0.5;\n"
        "                    const style = value1 === null ? '' : 'background:' + heatColor(level) + ';color:' + (level < 0.49 ? 'white' : '#101722');\n"
        "                    const display = value => value === null ? '—' : value.toFixed(3);\n"
        "                    markup += '<button type=\\\"button\\\" class=\\\"heat-cell forced-action-cell' + (actionIndex === selectedForcedAction ? ' selected' : '') + '\\\" data-action-index=\\\"' + actionIndex + '\\\" aria-pressed=\\\"' + (actionIndex === selectedForcedAction) + '\\\" style=\\\"' + style + '\\\" title=\\\"' + label + '\\\" aria-label=\\\"' + label + '\\\">' + '<span>' + buttonLabel0 + '</span><span>' + display(value0) + '</span><span>' + buttonLabel1 + '</span><span>' + display(value1) + '</span><span>' + buttonLabel2 + '</span><span>' + display(value2) + '</span>' + '</button>';\n"
        "                }\n"
        "            }\n"
        "            grid.innerHTML = markup;\n"
        "            grid.querySelectorAll('[data-action-index]').forEach(button => button.addEventListener('click', () => {\n"
        "                const actionIndex = Number(button.dataset.actionIndex);\n"
        "                const currentStep = step;\n"
        "                selectedForcedAction = actionIndex; buildForcedActionSelector();\n"
        "                initReplay(actionIndex, currentStep).catch(showReplayError);\n"
        "            }));\n"
        "        }\n"
        "        function showReplayError(error) {\n"
        "            console.error(error);\n"
        "            replayLoading = false;\n"
        "            document.getElementById('load-text').textContent = 'Replay load failed. See console.';\n"
        "            document.getElementById('loading-overlay').style.display = 'flex';\n"
        "        }\n"
        "        function heatColor(t) {",
        "standard heat color function",
    )
    html = _replace_once(
        html,
        "        initReplay().catch(err => { console.error(err); document.getElementById('load-text').textContent = 'Replay load failed. See console.'; });",
        "        initReplay(0).catch(showReplayError);",
        "initial replay load",
    )
    return html


def _add_comparison_view(html):
    html = _replace_once(
        html,
        '<button type="button" id="camMode" class="cam-chip" onclick="toggleCamMode()">world cam</button>',
        '<button type="button" id="compare-toggle" class="cam-chip comparison-toggle" onclick="toggleComparison()" aria-label="Compare all 12 forced actions">compare 12</button>'
        '<button type="button" id="camMode" class="cam-chip" onclick="toggleCamMode()">world cam</button>',
        "world camera control",
    )
    comparison_styles = """
        .comparison-toggle + #camMode { margin-left:0; }
        #comparison-toolbar { position:absolute; z-index:20; top:10px; left:50%; transform:translateX(-50%); display:none; gap:10px; align-items:center; padding:7px 10px; white-space:nowrap; }
        .comparison-toolbar-label { color:var(--muted); font-size:11px; font-weight:600; }
        #comparison-grid { position:absolute; z-index:1; inset:52px 10px 68px; display:none; grid-template-columns:repeat(3,minmax(0,1fr)); grid-template-rows:repeat(4,minmax(0,1fr)); gap:4px; pointer-events:none; }
        .comparison-cell { position:relative; min-width:0; min-height:0; overflow:hidden; border:1px solid var(--border); border-radius:5px; background:var(--bg); }
        .comparison-cell[data-ended="true"] { border-color:var(--muted); }
        .comparison-tile { display:block; width:100%; height:100%; cursor:pointer; pointer-events:auto; }
        .comparison-label { position:absolute; top:0; left:0; right:0; z-index:1; display:flex; justify-content:space-between; gap:4px; padding:3px 6px; background:rgba(13,15,18,.78); color:#fff; font:600 10px/1.3 system-ui,sans-serif; pointer-events:none; }
        .comparison-frame { white-space:nowrap; color:#d9e4f2; }
        body.comparison-active #c { display:none; }
        body.comparison-active #hud-global, body.comparison-active #hud-telemetry, body.comparison-active #obs-container { display:none !important; }
        body.comparison-active #comparison-grid { display:grid; }
        body.comparison-active #comparison-toolbar { display:flex; }
        @media (max-width:760px) {
            #comparison-grid { inset:48px 5px 62px; gap:2px; }
            .comparison-label { padding:2px 3px; font-size:8px; }
            #comparison-toolbar { top:6px; }
            .comparison-toolbar-label { display:none; }
        }
    """
    html = _replace_once(html, "    </style>", comparison_styles + "    </style>", "standard viewer style terminator")
    html = _replace_once(
        html,
        '        </div>\n    </div>\n    <canvas id="c"></canvas>',
        '        </div>\n'
        '        <div id="comparison-toolbar" class="panel" role="group" aria-label="Action comparison controls">'
        '<span class="comparison-toolbar-label">All actions share the same frame; ended branches stay at their last frame.</span>'
        '<button type="button" id="comparison-exit" class="btn" onclick="exitComparison(selectedForcedAction)">Return to selected action</button></div>\n'
        '        <div id="comparison-grid" aria-label="Forced action comparison"></div>\n'
        '    </div>\n    <canvas id="c"></canvas>',
        "comparison canvas and toolbar",
    )

    html = _replace_once(
        html,
        "        function frameMax() { return Math.max(0, (H ? H.frames : 1) - 1); }",
        "        function frameMax() {\n"
        "            if (comparisonRendering && comparisonRenderBranch) return comparisonRenderBranch.maxFrame;\n"
        "            if (comparisonActive && comparisonBranches.length) return comparisonLongestFrame();\n"
        "            return Math.max(0, (H ? H.frames : 1) - 1);\n"
        "        }",
        "standard frame limit",
    )
    html = _replace_once(
        html,
        "        window.onresize = () => { c.width = innerWidth; c.height = innerHeight; resizeObsCanvas(); draw(true); };",
        "        window.onresize = () => {\n"
        "            if (comparisonActive) { resizeObsCanvas(); draw(true); return; }\n"
        "            c.width = innerWidth; c.height = innerHeight; resizeObsCanvas(); draw(true);\n"
        "        };",
        "standard viewer resize handler",
    )
    html = _replace_once(
        html,
        "        function selectedGoals(frame, agent) {\n            if (!C.obs || !agent || agent.slot < 0) return [];",
        "        function selectedGoals(frame, agent) {\n"
        "            if (C.comparison_goal_points && C.comparison_goal_valid && agent) {\n"
        "                const out = [];\n"
        "                for (let i = 0; i < H.num_goals; i++) {\n"
        "                    const goalOffset = frame * H.num_goals + i;\n"
        "                    if (!C.comparison_goal_valid[goalOffset]) continue;\n"
        "                    out.push({x:C.comparison_goal_points[goalOffset * 2], y:C.comparison_goal_points[goalOffset * 2 + 1]});\n"
        "                }\n"
        "                return out;\n"
        "            }\n"
        "            if (!C.obs || !agent || agent.slot < 0) return [];",
        "standard replay goal rendering",
    )
    html = _replace_once(
        html,
        "        function draw(force=false) {\n            if(!H) return;",
        "        function draw(force=false) {\n"
        "            if (comparisonActive && !comparisonRendering) { drawComparisonFrame(force); return; }\n"
        "            if(!H) return;",
        "standard draw entry",
    )
    html = _replace_once(
        html,
        "            updateUI(target);\n            const colors = getColors();",
        "            if (!comparisonRendering) updateUI(target);\n            const colors = getColors();",
        "standard replay HUD update",
    )
    html = _replace_once(
        html,
        "        function toggle(){ play=!play; lastTick=performance.now(); updateBtn(); if(play) requestAnimationFrame(loop); }",
        "        function toggle(){\n"
        "            if (comparisonActive && step >= frameMax()) { step = 0; draw(true); }\n"
        "            play=!play; lastTick=performance.now(); updateBtn(); if(play) requestAnimationFrame(loop);\n"
        "        }",
        "standard playback toggle",
    )
    html = _replace_once(
        html,
        "            step += dt * speed * 10;\n            while(step > frameMax()) step -= frameMax() + 1;\n            draw();",
        "            step += dt * speed * 10;\n"
        "            if (comparisonActive) {\n"
        "                if (step >= frameMax()) { step = frameMax(); play = false; updateBtn(); }\n"
        "            } else {\n"
        "                while(step > frameMax()) step -= frameMax() + 1;\n"
        "            }\n"
        "            draw();",
        "standard playback loop",
    )

    comparison_script = Path(__file__).with_name("comparison_view.js").read_text(encoding="utf-8")
    return _replace_once(
        html,
        "        initReplay(0).catch(showReplayError);",
        comparison_script + "\n        initReplay(0).catch(showReplayError);",
        "standard replay initialization call",
    )


def _replace_once(text, old, new, description):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"standard SAC viewer changed around {description}: expected one match, found {count}")
    return text.replace(old, new, 1)


def _replace_regex_once(text, pattern, replacement, description):
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise RuntimeError(f"standard SAC viewer changed around {description}: expected one match, found {len(matches)}")
    match = matches[0]
    return text[:match.end()] + "\n" + replacement + text[match.end():]
