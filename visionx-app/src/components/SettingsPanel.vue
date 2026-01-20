<script setup lang="ts">
import { ref } from "vue";

interface Settings {
  confidence: number;
  stride: number;
  halfPrecision: boolean;
  outputDir: string;
  searchPrompts: string[];
}

const settings = defineModel<Settings>({ required: true });

const isExpanded = ref(false);
const newPrompt = ref("");

function addPrompt() {
  if (newPrompt.value.trim()) {
    settings.value.searchPrompts.push(newPrompt.value.trim());
    newPrompt.value = "";
  }
}

function removePrompt(index: number) {
  settings.value.searchPrompts.splice(index, 1);
}
</script>

<template>
  <div class="settings-panel card">
    <button class="toggle-btn" @click="isExpanded = !isExpanded">
      <span>Settings</span>
      <span class="arrow" :class="{ expanded: isExpanded }">▼</span>
    </button>

    <div v-if="isExpanded" class="settings-content">
      <!-- Confidence -->
      <div class="setting-row">
        <label>
          <span class="label-text">Confidence Threshold</span>
          <span class="label-value">{{ settings.confidence.toFixed(2) }}</span>
        </label>
        <input
          type="range"
          v-model.number="settings.confidence"
          min="0"
          max="1"
          step="0.05"
        />
      </div>

      <!-- Stride -->
      <div class="setting-row">
        <label>
          <span class="label-text">Frame Stride</span>
          <span class="label-value">Every {{ settings.stride }} frame(s)</span>
        </label>
        <input
          type="range"
          v-model.number="settings.stride"
          min="1"
          max="10"
          step="1"
        />
      </div>

      <!-- Half Precision -->
      <div class="setting-row checkbox">
        <label>
          <input type="checkbox" v-model="settings.halfPrecision" />
          <span class="label-text">Half Precision (FP16) - Faster processing</span>
        </label>
      </div>

      <!-- Custom Search Prompts -->
      <div class="setting-row">
        <label>
          <span class="label-text">Custom Search Prompts (optional)</span>
        </label>
        <div class="prompts-input">
          <input
            type="text"
            v-model="newPrompt"
            placeholder="e.g., white car, red motorcycle"
            @keyup.enter="addPrompt"
          />
          <button class="secondary" @click="addPrompt">Add</button>
        </div>
        <div v-if="settings.searchPrompts.length > 0" class="prompts-list">
          <span
            v-for="(prompt, index) in settings.searchPrompts"
            :key="index"
            class="prompt-tag"
          >
            {{ prompt }}
            <button @click="removePrompt(index)">×</button>
          </span>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.settings-panel {
  padding: 0;
  overflow: hidden;
}

.toggle-btn {
  width: 100%;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 20px;
  background: transparent;
  color: var(--text-primary);
  font-size: 1rem;
  font-weight: 500;
}

.toggle-btn:hover {
  background: var(--bg-secondary);
}

.arrow {
  font-size: 0.8rem;
  transition: transform 0.2s ease;
}

.arrow.expanded {
  transform: rotate(180deg);
}

.settings-content {
  padding: 0 20px 20px;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.setting-row {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.setting-row label {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.label-text {
  font-size: 0.9rem;
}

.label-value {
  font-size: 0.85rem;
  color: var(--accent);
}

.setting-row.checkbox label {
  flex-direction: row;
  gap: 10px;
  cursor: pointer;
}

.setting-row.checkbox input {
  width: 18px;
  height: 18px;
}

input[type="range"] {
  width: 100%;
  height: 6px;
  border-radius: 3px;
  background: var(--bg-secondary);
  -webkit-appearance: none;
}

input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: var(--accent);
  cursor: pointer;
}

.prompts-input {
  display: flex;
  gap: 8px;
}

.prompts-input input {
  flex: 1;
}

.prompts-list {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 8px;
}

.prompt-tag {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  background: var(--bg-secondary);
  border-radius: 20px;
  font-size: 0.85rem;
}

.prompt-tag button {
  background: transparent;
  color: var(--text-secondary);
  padding: 0;
  font-size: 1rem;
  line-height: 1;
}

.prompt-tag button:hover {
  color: var(--accent);
}
</style>
