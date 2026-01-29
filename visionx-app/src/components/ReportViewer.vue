<script setup lang="ts">
import { ref, computed, watch } from "vue";
import { invoke } from "@tauri-apps/api/core";

const props = defineProps<{
  reports: string[];
}>();

const selected = defineModel<string | null>("selected");
const reportContent = ref<string>("");
const loadingPreview = ref(false);

const hasReports = computed(() => props.reports.length > 0);

// Get just the filename from a path
function getFileName(path: string): string {
  return path.split('/').pop() || path;
}

// Watch for selected report changes and load preview
watch(selected, async (newPath) => {
  if (newPath) {
    loadingPreview.value = true;
    try {
      const content = await invoke<string>('get_report_content', { path: newPath });
      reportContent.value = content;
    } catch (e) {
      console.error("Failed to load report:", e);
      reportContent.value = "";
    } finally {
      loadingPreview.value = false;
    }
  } else {
    reportContent.value = "";
  }
}, { immediate: true });

async function openInBrowser(report: string) {
  try {
    await invoke('open_file', { path: report });
  } catch (e) {
    console.error("Failed to open report:", e);
  }
}

async function openFolder() {
  if (props.reports.length > 0) {
    try {
      await invoke('show_in_folder', { path: props.reports[0] });
    } catch (e) {
      console.error("Failed to open folder:", e);
    }
  }
}

function selectReport(report: string) {
  selected.value = report;
}
</script>

<template>
  <div class="report-viewer">
    <div class="reports-list card">
      <h3>Αναφορές ({{ reports.length }})</h3>
      <ul v-if="hasReports">
        <li
          v-for="report in reports"
          :key="report"
          :class="{ active: selected === report }"
          @click="selectReport(report)"
        >
          <span class="report-name">{{ getFileName(report) }}</span>
        </li>
      </ul>
      <p v-else class="no-reports">Δεν έχουν δημιουργηθεί αναφορές ακόμα.</p>

    </div>

    <div v-if="selected" class="report-preview card">
      <div class="preview-header">
        <h3>Προεπισκόπηση: {{ getFileName(selected) }}</h3>
        <div class="preview-actions">
          <button class="secondary" @click="openFolder">
            Άνοιγμα Φακέλου
          </button>
          <button class="secondary" @click="openInBrowser(selected)">
            Άνοιγμα στον Browser
          </button>
        </div>
      </div>
      <div class="preview-frame">
        <div v-if="loadingPreview" class="placeholder">
          <p>Φόρτωση προεπισκόπησης...</p>
        </div>
        <iframe
          v-else-if="reportContent"
          :srcdoc="reportContent"
          class="report-iframe"
        ></iframe>
        <div v-else class="placeholder">
          <p>Αδυναμία φόρτωσης προεπισκόπησης.</p>
          <p class="hint">Πατήστε "Άνοιγμα στον Browser" για πλήρη λειτουργικότητα.</p>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.report-viewer {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.reports-list h3 {
  margin-bottom: 16px;
  font-size: 1rem;
}

.reports-list ul {
  list-style: none;
  max-height: 200px;
  overflow-y: auto;
}

.reports-list li {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 14px;
  background: var(--bg-secondary);
  border-radius: 6px;
  margin-bottom: 8px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.reports-list li:hover {
  background: var(--bg-primary);
}

.reports-list li.active {
  border: 1px solid var(--accent);
}

.report-name {
  font-size: 0.9rem;
}

.no-reports {
  color: var(--text-secondary);
  text-align: center;
  padding: 20px;
}

.report-preview {
  display: flex;
  flex-direction: column;
}

.preview-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.preview-header h3 {
  font-size: 0.95rem;
}

.preview-actions {
  display: flex;
  gap: 8px;
}

.preview-frame {
  background: var(--bg-secondary);
  border-radius: 8px;
  min-height: 400px;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}

.report-iframe {
  width: 100%;
  height: 400px;
  border: none;
  border-radius: 8px;
  background: white;
}

.placeholder {
  text-align: center;
  color: var(--text-secondary);
}

.placeholder .hint {
  font-size: 0.85rem;
  margin-top: 8px;
}
</style>
