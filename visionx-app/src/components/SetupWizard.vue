<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from "vue";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import type { UnlistenFn } from "@tauri-apps/api/event";

const emit = defineEmits<{
  (e: "complete"): void;
}>();

interface GpuInfo {
  has_nvidia: boolean;
  gpu_name: string;
  driver_version: string;
}

interface SetupProgressEvent {
  step: string;
  step_label: string;
  downloaded: number;
  total: number;
  step_index: number;
  total_steps: number;
}

// State
const phase = ref<"detecting" | "ready" | "installing" | "complete" | "error">("detecting");
const gpuInfo = ref<GpuInfo>({ has_nvidia: false, gpu_name: "", driver_version: "" });
const errorMessage = ref("");
const logPath = ref("");

// Progress tracking
const currentStep = ref("");
const currentStepLabel = ref("");
const downloaded = ref(0);
const total = ref(0);
const stepIndex = ref(0);
const totalSteps = ref(5);
const downloadSpeed = ref(0);

// Speed calculation
let lastDownloaded = 0;
let lastSpeedTime = Date.now();

const steps = computed(() => [
  { key: "python", label: "Python 3.13", size: "~25 MB" },
  { key: "pytorch", label: gpuInfo.value.has_nvidia ? "PyTorch (CUDA)" : "PyTorch (CPU)", size: gpuInfo.value.has_nvidia ? "~2.2 GB" : "~590 MB" },
  { key: "deps", label: "AI Libraries", size: "~150 MB" },
  { key: "model", label: "AI Model", size: "~171 MB" },
  { key: "scripts", label: "Finalization", size: "" },
]);

const currentStepStatus = (stepKey: string) => {
  const idx = steps.value.findIndex((s) => s.key === stepKey);
  const current = steps.value.findIndex((s) => s.key === currentStep.value);
  if (idx < current) return "done";
  if (idx === current) return "active";
  return "pending";
};

const overallPercent = computed(() => {
  if (totalSteps.value === 0) return 0;
  const stepProgress = total.value > 0 ? downloaded.value / total.value : 0;
  return Math.round(((stepIndex.value - 1 + stepProgress) / totalSteps.value) * 100);
});

const formatSize = (bytes: number) => {
  if (bytes < 1_000_000) return `${(bytes / 1_000).toFixed(0)} KB`;
  if (bytes < 1_000_000_000) return `${(bytes / 1_000_000).toFixed(1)} MB`;
  return `${(bytes / 1_000_000_000).toFixed(2)} GB`;
};

const speedText = computed(() => {
  if (downloadSpeed.value <= 0) return "";
  return `${formatSize(downloadSpeed.value)}/s`;
});

const etaText = computed(() => {
  if (downloadSpeed.value <= 0 || total.value <= 0) return "";
  const remaining = total.value - downloaded.value;
  const seconds = Math.round(remaining / downloadSpeed.value);
  if (seconds < 60) return `~${seconds}s`;
  if (seconds < 3600) return `~${Math.round(seconds / 60)}m`;
  return `~${Math.floor(seconds / 3600)}h ${Math.round((seconds % 3600) / 60)}m`;
});

let unlisten: UnlistenFn | null = null;

onMounted(async () => {
  // Get log path
  try {
    logPath.value = await invoke("get_log_path");
  } catch (_) {}

  // Listen for progress events
  unlisten = await listen<SetupProgressEvent>("setup-progress", (event) => {
    const p = event.payload;
    currentStep.value = p.step;
    currentStepLabel.value = p.step_label;
    downloaded.value = p.downloaded;
    total.value = p.total;
    stepIndex.value = p.step_index;
    totalSteps.value = p.total_steps;

    // Calculate speed
    const now = Date.now();
    const elapsed = (now - lastSpeedTime) / 1000;
    if (elapsed >= 1) {
      downloadSpeed.value = (p.downloaded - lastDownloaded) / elapsed;
      lastDownloaded = p.downloaded;
      lastSpeedTime = now;
    }
  });

  // Detect GPU
  try {
    gpuInfo.value = await invoke("detect_gpu");
    phase.value = "ready";
  } catch (e) {
    gpuInfo.value = { has_nvidia: false, gpu_name: "", driver_version: "" };
    phase.value = "ready";
  }
});

onUnmounted(() => {
  if (unlisten) unlisten();
});

async function startSetup() {
  phase.value = "installing";
  lastDownloaded = 0;
  lastSpeedTime = Date.now();
  downloadSpeed.value = 0;

  try {
    await invoke("run_setup", { useCuda: gpuInfo.value.has_nvidia });
    phase.value = "complete";
    // Auto-proceed after a moment
    setTimeout(() => emit("complete"), 1500);
  } catch (e: any) {
    phase.value = "error";
    errorMessage.value = String(e);
  }
}

async function openLog() {
  if (logPath.value) {
    try {
      await invoke("open_file", { path: logPath.value });
    } catch (_) {}
  }
}

async function retry() {
  errorMessage.value = "";
  phase.value = "ready";
}
</script>

<template>
  <div class="setup-container card">
    <h1 class="title">VisionX Setup</h1>

    <!-- GPU Detection -->
    <div class="gpu-info" :class="{ nvidia: gpuInfo.has_nvidia }">
      <template v-if="phase === 'detecting'">
        <div class="spinner-small"></div>
        <span>Detecting hardware...</span>
      </template>
      <template v-else-if="gpuInfo.has_nvidia">
        <span class="gpu-icon">GPU</span>
        <div>
          <div class="gpu-name">{{ gpuInfo.gpu_name }}</div>
          <div class="gpu-detail">CUDA {{ gpuInfo.driver_version }} &mdash; GPU acceleration enabled</div>
        </div>
      </template>
      <template v-else>
        <span class="gpu-icon cpu">CPU</span>
        <div>
          <div class="gpu-name">No NVIDIA GPU detected</div>
          <div class="gpu-detail">CPU mode &mdash; slower but functional</div>
        </div>
      </template>
    </div>

    <!-- Ready state: show what will be installed -->
    <template v-if="phase === 'ready'">
      <p class="description">
        Required components need to be downloaded for first-time use.
        This will take a few minutes depending on your internet speed.
      </p>
      <div class="size-estimate">
        Total download: <strong>{{ gpuInfo.has_nvidia ? '~2.6 GB' : '~1 GB' }}</strong>
      </div>
      <button class="btn-primary" @click="startSetup">Start Setup</button>
    </template>

    <!-- Installing: show progress -->
    <template v-if="phase === 'installing'">
      <div class="steps-list">
        <div
          v-for="step in steps"
          :key="step.key"
          class="step-row"
          :class="currentStepStatus(step.key)"
        >
          <span class="step-icon">
            <template v-if="currentStepStatus(step.key) === 'done'">&#10003;</template>
            <template v-else-if="currentStepStatus(step.key) === 'active'">
              <div class="spinner-tiny"></div>
            </template>
            <template v-else>&#9675;</template>
          </span>
          <span class="step-label">{{ step.label }}</span>
          <span class="step-size">
            <template v-if="currentStepStatus(step.key) === 'active' && total > 0">
              {{ formatSize(downloaded) }} / {{ formatSize(total) }}
            </template>
            <template v-else-if="currentStepStatus(step.key) === 'active' && total === 0">
              Εγκατάσταση...
            </template>
            <template v-else-if="currentStepStatus(step.key) === 'done'">
              OK
            </template>
            <template v-else>
              {{ step.size }}
            </template>
          </span>
        </div>
      </div>

      <!-- Overall progress bar -->
      <div class="progress-section">
        <div class="progress-bar">
          <div class="progress-fill" :style="{ width: overallPercent + '%' }"></div>
        </div>
        <div class="progress-stats">
          <span>{{ overallPercent }}%</span>
          <span v-if="speedText">{{ speedText }}</span>
          <span v-if="etaText">{{ etaText }}</span>
        </div>
      </div>

    </template>

    <!-- Complete -->
    <template v-if="phase === 'complete'">
      <div class="complete-message">
        <span class="check-large">&#10003;</span>
        <p>Setup complete! Starting VisionX...</p>
      </div>
    </template>

    <!-- Error -->
    <template v-if="phase === 'error'">
      <div class="error-section">
        <p class="error-title">Setup failed</p>
        <p class="error-message">{{ errorMessage }}</p>
        <div class="error-actions">
          <button class="btn-secondary" @click="openLog">View Log</button>
          <button class="btn-primary" @click="retry">Retry</button>
        </div>
      </div>
    </template>
  </div>
</template>

<style scoped>
.setup-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 24px;
  max-width: 500px;
  margin: 0 auto;
  padding: 40px 32px;
}

.title {
  font-size: 1.5rem;
  font-weight: 600;
  margin: 0;
}

.description {
  text-align: center;
  color: var(--text-secondary);
  font-size: 0.9rem;
  line-height: 1.5;
  margin: 0;
}

.size-estimate {
  color: var(--text-secondary);
  font-size: 0.85rem;
}

/* GPU Info */
.gpu-info {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 20px;
  border-radius: 8px;
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  width: 100%;
}

.gpu-info.nvidia {
  border-color: var(--success);
  background: rgba(76, 175, 80, 0.08);
}

.gpu-icon {
  padding: 4px 10px;
  border-radius: 4px;
  font-size: 0.75rem;
  font-weight: 700;
  background: var(--success);
  color: white;
  flex-shrink: 0;
}

.gpu-icon.cpu {
  background: var(--text-secondary);
}

.gpu-name {
  font-weight: 500;
  font-size: 0.9rem;
}

.gpu-detail {
  font-size: 0.8rem;
  color: var(--text-secondary);
}

/* Steps list */
.steps-list {
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.step-row {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 12px;
  border-radius: 6px;
  font-size: 0.9rem;
}

.step-row.active {
  background: rgba(52, 152, 219, 0.08);
}

.step-row.done {
  color: var(--text-secondary);
}

.step-icon {
  width: 20px;
  text-align: center;
  flex-shrink: 0;
}

.step-row.done .step-icon {
  color: var(--success);
  font-weight: 700;
}

.step-label {
  flex: 1;
}

.step-size {
  font-size: 0.8rem;
  color: var(--text-secondary);
  text-align: right;
}

.step-row.done .step-size {
  color: var(--success);
}

/* Progress */
.progress-section {
  width: 100%;
}

.progress-bar {
  height: 6px;
  background: var(--bg-secondary);
  border-radius: 3px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: var(--accent);
  border-radius: 3px;
  transition: width 0.3s ease;
}

.progress-stats {
  display: flex;
  justify-content: space-between;
  font-size: 0.8rem;
  color: var(--text-secondary);
  margin-top: 6px;
}

/* Buttons */
.btn-primary {
  padding: 12px 40px;
  background: var(--accent);
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 1rem;
  font-weight: 500;
  cursor: pointer;
  transition: opacity 0.2s;
}

.btn-primary:hover {
  opacity: 0.9;
}

.btn-secondary {
  padding: 8px 20px;
  background: transparent;
  color: var(--text-secondary);
  border: 1px solid var(--border);
  border-radius: 6px;
  font-size: 0.85rem;
  cursor: pointer;
}

.btn-secondary:hover {
  background: var(--bg-secondary);
}

/* Complete */
.complete-message {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
}

.check-large {
  font-size: 3rem;
  color: var(--success);
}

/* Error */
.error-section {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  width: 100%;
}

.error-title {
  font-weight: 600;
  color: var(--error, #e74c3c);
  margin: 0;
}

.error-message {
  font-size: 0.85rem;
  color: var(--text-secondary);
  text-align: center;
  word-break: break-word;
  max-height: 100px;
  overflow-y: auto;
  margin: 0;
}

.error-actions {
  display: flex;
  gap: 12px;
}

/* Spinners */
.spinner-small {
  width: 20px;
  height: 20px;
  border: 2px solid var(--border);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

.spinner-tiny {
  width: 14px;
  height: 14px;
  border: 2px solid var(--border);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}
</style>
