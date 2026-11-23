// Inlined into the standard SAC replay script so comparison frames use its renderer.
const COMPARISON_ACTION_COUNT = 12;
const COMPARISON_DRAW_CHUNKS = [
    "agent_f32", "agent_i32", "metrics_f32", "traffic_i16",
    "traffic_stop_lines", "traffic_types", "ghost_f32",
];
let comparisonActive = false;
let comparisonLoading = false;
let comparisonRendering = false;
let comparisonRenderBranch = null;
let comparisonBranches = [];
let comparisonSavedState = null;
let comparisonLastDrawnFrame = -1;
let comparisonLastDrawnWidth = -1;
let comparisonLastDrawnHeight = -1;

function comparisonTypedView(buffer, dataStart, metadata) {
    const type = metadata.dtype === "float32" ? Float32Array
        : metadata.dtype === "int32" ? Int32Array
        : metadata.dtype === "int16" ? Int16Array : Uint8Array;
    const itemBytes = type.BYTES_PER_ELEMENT;
    return new type(buffer, dataStart + metadata.offset, metadata.nbytes / itemBytes);
}

function comparisonGoalPoints(header, observations, agentFloat, agentInteger) {
    if (!observations || !header.obs_dim || !header.num_goals) return null;
    const goalPoints = new Float32Array(header.frames * header.num_goals * 2);
    const goalValid = new Uint8Array(header.frames * header.num_goals);
    const quantization = header.obs_scale === undefined ? 1 : header.obs_scale;
    const goalScale = header.scales.obs_norm_goal_offset_m * quantization;

    for (let frame = 0; frame < header.frames; frame++) {
        let egoIndex = -1;
        for (let agentIndex = 0; agentIndex < header.agent_cap; agentIndex++) {
            const integerBase = (frame * header.agent_cap + agentIndex) * header.chunks.agent_i32.shape[2];
            if (agentInteger[integerBase + 2] && agentInteger[integerBase] === 0) {
                egoIndex = agentIndex;
                break;
            }
        }
        if (egoIndex < 0) continue;

        const integerBase = (frame * header.agent_cap + egoIndex) * header.chunks.agent_i32.shape[2];
        const slot = agentInteger[integerBase + 7];
        if (slot < 0 || slot >= header.active_count) continue;
        const agentBase = (frame * header.agent_cap + egoIndex) * header.chunks.agent_f32.shape[2];
        const x = agentFloat[agentBase];
        const y = agentFloat[agentBase + 1];
        const heading = agentFloat[agentBase + 3];
        const observationBase = (frame * header.active_count + slot) * header.obs_dim;
        let goalsBase = observationBase + header.ego_dim;
        if (header.reward_conditioning) goalsBase += header.reward_coef_count;

        for (let goalIndex = 0; goalIndex < header.num_goals; goalIndex++) {
            const goalObservation = goalsBase + goalIndex * header.goal_features;
            let populated = false;
            for (let feature = 0; feature < header.goal_features; feature++) {
                if (observations[goalObservation + feature] !== 0) populated = true;
            }
            if (!populated) continue;
            const relativeX = observations[goalObservation] * goalScale;
            const relativeY = observations[goalObservation + 1] * goalScale;
            const goalOffset = frame * header.num_goals + goalIndex;
            goalPoints[goalOffset * 2] = x + relativeX * Math.cos(heading) - relativeY * Math.sin(heading);
            goalPoints[goalOffset * 2 + 1] = y + relativeX * Math.sin(heading) + relativeY * Math.cos(heading);
            goalValid[goalOffset] = 1;
        }
    }
    return {goalPoints, goalValid};
}

function comparisonBranch(actionIndex, header, sourceChunks) {
    const branchHeader = {...header, chunks: {...header.chunks}, buffer: null, dataStart: 0};
    const branchChunks = {};
    for (const name of COMPARISON_DRAW_CHUNKS) {
        if (sourceChunks[name]) branchChunks[name] = sourceChunks[name].slice();
    }
    const goals = comparisonGoalPoints(header, sourceChunks.obs, branchChunks.agent_f32, branchChunks.agent_i32);
    if (goals) {
        branchChunks.comparison_goal_points = goals.goalPoints;
        branchChunks.comparison_goal_valid = goals.goalValid;
    }
    const features = {
        af: branchHeader.chunks.agent_f32.shape[2],
        ai: branchHeader.chunks.agent_i32.shape[2],
        mf: branchHeader.chunks.metrics_f32.shape[2],
        tf: branchHeader.chunks.traffic_i16.shape[2],
    };
    return {
        actionIndex,
        header: branchHeader,
        chunks: branchChunks,
        features,
        expertIndices: new Set(branchHeader.expert_indices || []),
        hasGhost: !!branchChunks.ghost_f32,
        maxFrame: Math.max(0, branchHeader.frames - 1),
    };
}

function comparisonBranchFromBuffer(actionIndex, buffer) {
    const view = new DataView(buffer);
    const headerLength = view.getUint32(0, true);
    const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 4, headerLength)));
    const dataStart = 4 + headerLength + ((-(4 + headerLength)) & 3);
    const sourceChunks = {};
    for (const name of [...COMPARISON_DRAW_CHUNKS, "obs"]) {
        if (header.chunks[name]) sourceChunks[name] = comparisonTypedView(buffer, dataStart, header.chunks[name]);
    }
    return comparisonBranch(actionIndex, header, sourceChunks);
}

function buildComparisonTiles() {
    const grid = document.getElementById("comparison-grid");
    const longValues = [-15, -4, 0, 4], lateralValues = [-4, 0, 4];
    let markup = "";
    for (let actionIndex = 0; actionIndex < COMPARISON_ACTION_COUNT; actionIndex++) {
        const longValue = longValues[Math.floor(actionIndex / lateralValues.length)];
        const lateralValue = lateralValues[actionIndex % lateralValues.length];
        markup += `<div class="comparison-cell" data-action-index="${actionIndex}" data-ended="false">`
            + `<canvas class="comparison-tile" data-action-index="${actionIndex}" aria-label="Action ${actionIndex}, jerk_long ${longValue}, jerk_lat ${lateralValue}"></canvas>`
            + `<div class="comparison-label"><span>Action ${actionIndex} · jerk ${longValue} / ${lateralValue}</span>`
            + `<span id="comparison-frame-${actionIndex}" class="comparison-frame"></span></div></div>`;
    }
    grid.innerHTML = markup;
    grid.querySelectorAll(".comparison-tile").forEach(canvas => canvas.addEventListener("click", () => {
        exitComparison(Number(canvas.dataset.actionIndex));
    }));
}

function comparisonLongestFrame() {
    return comparisonBranches.reduce((longest, branch) => Math.max(longest, branch.maxFrame), 0);
}

function comparisonState() {
    const frame = Math.max(0, Math.floor(step));
    return {
        active: comparisonActive,
        loading: comparisonLoading,
        columns: 3,
        rows: 4,
        sync: true,
        freezeTerminal: true,
        frame,
        longestFrame: comparisonActive ? comparisonLongestFrame() : 0,
        selectedAction: selectedForcedAction,
        restoreAction: comparisonSavedState ? comparisonSavedState.actionIndex : selectedForcedAction,
        branchFrames: comparisonBranches.map(branch => Math.min(frame, branch.maxFrame)),
        branchEndFrames: comparisonBranches.map(branch => branch.maxFrame),
        endedActions: comparisonBranches.filter(branch => frame >= branch.maxFrame).map(branch => branch.actionIndex),
    };
}
window.__qcalComparisonState = comparisonState;

async function toggleComparison() {
    if (comparisonActive) {
        await exitComparison(selectedForcedAction);
        return;
    }
    if (comparisonLoading || !H) return;
    await enterComparison();
}

async function enterComparison() {
    if (replayLoading) return;
    comparisonLoading = true;
    play = false;
    updateBtn();
    const overlay = document.getElementById("loading-overlay");
    overlay.style.display = "flex";
    document.getElementById("load-text").textContent = "Preparing 12 action branches...";
    const actionIndexAtEntry = selectedForcedAction;
    comparisonSavedState = {
        actionIndex: actionIndexAtEntry,
        H, C, F, paths, expertAgentIndices, showGhost, followedId, isEgoCam,
        cam: {...cam},
        step,
    };

    try {
        comparisonBranches = [];
        for (let actionIndex = 0; actionIndex < COMPARISON_ACTION_COUNT; actionIndex++) {
            document.getElementById("load-text").textContent = `Preparing action ${actionIndex + 1} / ${COMPARISON_ACTION_COUNT}...`;
            if (actionIndex === actionIndexAtEntry) {
                comparisonBranches.push(comparisonBranch(actionIndex, H, C));
                continue;
            }
            const buffer = await decodeReplayPayload(actionIndex);
            comparisonBranches.push(comparisonBranchFromBuffer(actionIndex, buffer));
        }
        comparisonActive = true;
        comparisonLoading = false;
        document.body.classList.add("comparison-active");
        comparisonLastDrawnFrame = -1;
        const longestFrame = comparisonLongestFrame();
        step = Math.max(0, Math.min(comparisonSavedState.step, longestFrame));
        document.getElementById("sld").max = longestFrame;
        document.getElementById("stepTotal").textContent = longestFrame;
        overlay.style.display = "none";
        requestAnimationFrame(() => draw(true));
    } catch (error) {
        console.error(error);
        comparisonBranches = [];
        comparisonLoading = false;
        comparisonSavedState = null;
        overlay.style.display = "none";
        document.getElementById("load-text").textContent = "Decoding replay...";
        showReplayError(error);
    }
}

async function exitComparison(actionIndex) {
    if (!comparisonActive) return;
    const savedState = comparisonSavedState;
    const selectedAction = Math.max(0, Math.min(COMPARISON_ACTION_COUNT - 1, actionIndex));
    const requestedStep = Math.max(0, Math.floor(step));
    comparisonActive = false;
    comparisonLoading = false;
    comparisonRendering = false;
    comparisonRenderBranch = null;
    play = false;
    updateBtn();
    document.body.classList.remove("comparison-active");
    comparisonBranches = [];
    comparisonSavedState = null;
    selectedForcedAction = selectedAction;
    c.width = Math.max(1, window.innerWidth);
    c.height = Math.max(1, window.innerHeight);
    lastDrawn = -1;

    if (savedState && savedState.actionIndex === selectedAction) {
        H = savedState.H;
        C = savedState.C;
        F = savedState.F;
        paths = savedState.paths;
        expertAgentIndices = savedState.expertAgentIndices;
        showGhost = savedState.showGhost;
        followedId = savedState.followedId;
        isEgoCam = savedState.isEgoCam;
        cam = {...savedState.cam};
        step = Math.min(requestedStep, frameMax());
        document.getElementById("sld").max = frameMax();
        document.getElementById("stepTotal").textContent = frameMax();
        document.getElementById("sld").value = step;
        document.getElementById("loading-overlay").style.display = "none";
        buildForcedActionSelector();
        draw(true);
        return;
    }

    document.getElementById("loading-overlay").style.display = "flex";
    document.getElementById("load-text").textContent = `Loading selected action ${selectedAction}...`;
    try {
        await initReplay(selectedAction, requestedStep);
    } catch (error) {
        showReplayError(error);
    }
}

function drawComparisonFrame(force = false) {
    if (!comparisonActive || !comparisonSavedState || comparisonBranches.length !== COMPARISON_ACTION_COUNT) return;
    const savedState = comparisonSavedState;
    const syncFrame = Math.max(0, Math.min(comparisonLongestFrame(), step));
    const width = Math.max(1, window.innerWidth);
    const canvases = [...document.querySelectorAll(".comparison-tile")];
    const gridRect = document.getElementById("comparison-grid").getBoundingClientRect();
    const viewportWidth = Math.max(1, Math.floor(gridRect.width / 3));
    const viewportHeight = Math.max(1, Math.floor(gridRect.height / 4));
    const renderWidth = Math.max(1, canvases[0].clientWidth || viewportWidth);
    const renderHeight = Math.max(1, canvases[0].clientHeight || viewportHeight);
    const renderedFrame = Math.floor(syncFrame);
    if (!force && renderedFrame === comparisonLastDrawnFrame
        && renderWidth === comparisonLastDrawnWidth && renderHeight === comparisonLastDrawnHeight) return;

    for (const branch of comparisonBranches) {
        const tile = canvases[branch.actionIndex];
        const tileWidth = Math.max(1, tile.clientWidth || viewportWidth);
        const tileHeight = Math.max(1, tile.clientHeight || viewportHeight);
        if (tile.width !== tileWidth) tile.width = tileWidth;
        if (tile.height !== tileHeight) tile.height = tileHeight;

        H = branch.header;
        C = branch.chunks;
        F = branch.features;
        expertAgentIndices = branch.expertIndices;
        showGhost = branch.hasGhost;
        followedId = 0;
        isEgoCam = savedState.isEgoCam;
        const zoom = Math.max(0.2, savedState.cam.z * tileWidth / width);
        cam = {x: savedState.cam.x, y: savedState.cam.y, z: zoom, drag: false, lx: 0, ly: 0};
        step = Math.min(syncFrame, branch.maxFrame);
        comparisonRenderBranch = branch;
        comparisonRendering = true;
        lastDrawn = -1;

        if (c.width !== tileWidth) c.width = tileWidth;
        if (c.height !== tileHeight) c.height = tileHeight;
        draw(true);
        const tileContext = tile.getContext("2d");
        tileContext.clearRect(0, 0, tile.width, tile.height);
        tileContext.drawImage(c, 0, 0, tile.width, tile.height);

        const ended = syncFrame >= branch.maxFrame;
        tile.closest(".comparison-cell").dataset.ended = String(ended);
        tile.dataset.frame = String(Math.min(Math.floor(syncFrame), branch.maxFrame));
        tile.dataset.ended = String(ended);
        const frameLabel = document.getElementById(`comparison-frame-${branch.actionIndex}`);
        frameLabel.textContent = ended ? `frame ${branch.maxFrame} · ended` : `frame ${Math.floor(syncFrame)}`;
    }

    H = savedState.H;
    C = savedState.C;
    F = savedState.F;
    paths = savedState.paths;
    expertAgentIndices = savedState.expertAgentIndices;
    showGhost = savedState.showGhost;
    followedId = savedState.followedId;
    isEgoCam = savedState.isEgoCam;
    cam = {...savedState.cam};
    step = syncFrame;
    comparisonRenderBranch = null;
    comparisonRendering = false;
    comparisonLastDrawnFrame = renderedFrame;
    comparisonLastDrawnWidth = renderWidth;
    comparisonLastDrawnHeight = renderHeight;
    document.getElementById("stepNow").textContent = String(Math.floor(syncFrame));
    document.getElementById("sld").max = comparisonLongestFrame();
    document.getElementById("sld").value = syncFrame;
    lastDrawn = -1;
}

buildComparisonTiles();
