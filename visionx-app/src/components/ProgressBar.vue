<script setup lang="ts">
import { computed } from "vue";

const props = defineProps<{
  currentVideo: string;
  currentFrame: number;
  totalFrames: number;
  videoIndex: number;
  totalVideos: number;
  fps: number;
}>();

const framePercent = computed(() => {
  if (props.totalFrames === 0) return 0;
  return Math.round((props.currentFrame / props.totalFrames) * 100);
});

const overallPercent = computed(() => {
  if (props.totalVideos === 0) return 0;
  if (props.totalFrames === 0) return 0;
  const completedVideos = props.videoIndex - 1;
  const currentProgress = props.currentFrame / props.totalFrames;
  return Math.round(((completedVideos + currentProgress) / props.totalVideos) * 100);
});

const videoName = computed(() => {
  return props.currentVideo.split("/").pop() || props.currentVideo;
});
</script>

<template>
  <div class="progress-container card">
    <div class="current-video">
      <span class="label">Processing:</span>
      <span class="video-name">{{ videoName }}</span>
    </div>

    <!-- Loading indicator when totalFrames is 0 -->
    <div v-if="totalFrames === 0" class="loading-section">
      <div class="spinner"></div>
      <p>Initializing... This may take a moment to load the model.</p>
      <p class="hint">The video will be analyzed after the model loads.</p>
    </div>

    <!-- Progress bars only shown when we have frame data -->
    <template v-else>
      <!-- Current video progress -->
      <div class="progress-section">
        <div class="progress-header">
          <span>Frame Progress</span>
          <span>{{ currentFrame }} / {{ totalFrames }} ({{ framePercent }}%)</span>
        </div>
        <div class="progress-bar">
          <div class="progress-fill" :style="{ width: framePercent + '%' }"></div>
        </div>
      </div>

      <!-- Overall progress -->
      <div class="progress-section">
        <div class="progress-header">
          <span>Overall Progress</span>
          <span>Video {{ videoIndex }} of {{ totalVideos }} ({{ overallPercent }}%)</span>
        </div>
        <div class="progress-bar overall">
          <div class="progress-fill" :style="{ width: overallPercent + '%' }"></div>
        </div>
      </div>

      <!-- Stats -->
      <div class="stats">
        <div class="stat">
          <span class="stat-value">{{ fps.toFixed(1) }}</span>
          <span class="stat-label">FPS</span>
        </div>
      </div>
    </template>
  </div>
</template>

<style scoped>
.progress-container {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.current-video {
  display: flex;
  align-items: center;
  gap: 8px;
}

.label {
  color: var(--text-secondary);
  font-size: 0.9rem;
}

.video-name {
  font-weight: 500;
  color: var(--accent);
}

.progress-section {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.progress-header {
  display: flex;
  justify-content: space-between;
  font-size: 0.85rem;
  color: var(--text-secondary);
}

.progress-bar {
  height: 8px;
  background: var(--bg-secondary);
  border-radius: 4px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: var(--accent);
  border-radius: 4px;
  transition: width 0.3s ease;
}

.progress-bar.overall .progress-fill {
  background: var(--success);
}

.stats {
  display: flex;
  justify-content: center;
  gap: 40px;
  padding-top: 10px;
  border-top: 1px solid var(--border);
}

.stat {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.stat-value {
  font-size: 1.5rem;
  font-weight: 600;
  color: var(--accent);
}

.stat-label {
  font-size: 0.75rem;
  color: var(--text-secondary);
  text-transform: uppercase;
}

.loading-section {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 40px 20px;
  text-align: center;
}

.spinner {
  width: 40px;
  height: 40px;
  border: 3px solid var(--border);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin-bottom: 16px;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.loading-section p {
  margin: 4px 0;
}

.loading-section .hint {
  font-size: 0.85rem;
  color: var(--text-secondary);
}
</style>
