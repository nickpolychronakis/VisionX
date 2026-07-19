<script setup lang="ts">
// About panel — replaces the old landing page (2026 desktop apps open
// straight into the workspace; identity/version/updates live behind ℹ️).
import { ref, onMounted } from "vue";
import { getVersion } from "@tauri-apps/api/app";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-shell";

interface UpdateInfo {
  available: boolean;
  version: string;
  current_version: string;
  download_url: string;
  can_auto_update: boolean;
}

const emit = defineEmits<{ close: [] }>();

const appVersion = ref("");
const updateInfo = ref<UpdateInfo | null>(null);
const checkingUpdate = ref(false);
const installingUpdate = ref(false);
const updateError = ref("");

onMounted(async () => {
  try {
    appVersion.value = await getVersion();
  } catch (e) {
    console.error("Failed to get version:", e);
  }
});

async function checkForUpdates() {
  checkingUpdate.value = true;
  updateError.value = "";
  updateInfo.value = null;
  try {
    updateInfo.value = await invoke<UpdateInfo>("check_for_updates");
  } catch (e) {
    updateError.value = String(e);
  } finally {
    checkingUpdate.value = false;
  }
}

async function installUpdate() {
  if (!updateInfo.value) return;
  if (updateInfo.value.can_auto_update) {
    installingUpdate.value = true;
    try {
      await invoke("install_update");
    } catch (e) {
      updateError.value = String(e);
    } finally {
      installingUpdate.value = false;
    }
  } else if (updateInfo.value.download_url) {
    await open(updateInfo.value.download_url);
  }
}

function dismissUpdate() {
  updateInfo.value = null;
  updateError.value = "";
}

async function openLog() {
  try {
    const logPath = await invoke<string>("get_log_path");
    await invoke("open_file", { path: logPath });
  } catch (e) {
    console.error("Failed to open log:", e);
  }
}
</script>

<template>
  <div class="about-backdrop" @click.self="emit('close')">
    <div class="about-panel card">
      <button class="close-btn" @click="emit('close')" title="Κλείσιμο">×</button>

      <img src="/logo.svg" alt="VisionX" class="about-logo" />
      <h2>VisionX</h2>
      <p class="about-tagline">
        Ανίχνευση &amp; παρακολούθηση αντικειμένων σε βίντεο ασφαλείας —
        100% τοπική εκτέλεση, τίποτα δεν αποστέλλεται σε servers.
      </p>

      <div class="dedication">
        Αφιερωμένο στα <strong>Γραφεία Προανακρίσεων</strong> των Τμημάτων
        Εξιχνιάσεων της Ελληνικής Αστυνομίας
      </div>

      <div class="about-meta">
        <p>Δημιουργήθηκε από <strong>Nick Polychronakis</strong></p>
        <p class="contact">
          <a href="mailto:nickpolychronakis@me.com">nickpolychronakis@me.com</a>
        </p>
        <p v-if="appVersion" class="version">Έκδοση {{ appVersion }}</p>
      </div>

      <div class="about-actions">
        <button class="secondary" @click="checkForUpdates" :disabled="checkingUpdate">
          {{ checkingUpdate ? "Έλεγχος..." : "Έλεγχος ενημερώσεων" }}
        </button>
        <button class="secondary" @click="openLog">Αρχείο Log</button>
      </div>

      <div v-if="updateInfo" class="update-note" :class="{ available: updateInfo.available }">
        <template v-if="updateInfo.available">
          <p><strong>Διαθέσιμη νέα έκδοση: {{ updateInfo.version }}</strong></p>
          <div class="about-actions">
            <button class="primary" @click="installUpdate" :disabled="installingUpdate">
              {{ installingUpdate ? "Εγκατάσταση..." : (updateInfo.can_auto_update ? "Εγκατάσταση" : "Λήψη") }}
            </button>
            <button class="secondary" @click="dismissUpdate">Αργότερα</button>
          </div>
        </template>
        <template v-else>
          <p>Έχετε την τελευταία έκδοση!</p>
          <button class="secondary" @click="dismissUpdate">OK</button>
        </template>
      </div>
      <div v-if="updateError" class="update-note error">
        <p>{{ updateError }}</p>
        <button class="secondary" @click="updateError = ''">OK</button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.about-backdrop {
  position: fixed;
  inset: 0;
  background: rgba(8, 10, 14, 0.55);
  backdrop-filter: blur(10px);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  animation: backdrop-in 0.2s ease;
}

.about-panel {
  position: relative;
  width: min(440px, 90vw);
  text-align: center;
  padding: 32px 28px 24px;
  animation: panel-in 0.24s cubic-bezier(0.2, 0.9, 0.3, 1.2);
}

.close-btn {
  position: absolute;
  top: 10px;
  right: 12px;
  background: transparent;
  color: var(--text-secondary);
  font-size: 1.5rem;
  padding: 2px 8px;
  line-height: 1;
}

.close-btn:hover {
  color: var(--text-primary);
}

.about-logo {
  width: 72px;
  height: 72px;
  margin-bottom: 8px;
}

.about-tagline {
  color: var(--text-secondary);
  font-size: 0.88rem;
  margin: 6px 0 14px;
  line-height: 1.5;
}

.dedication {
  font-size: 0.85rem;
  border: 1px solid rgba(79, 140, 255, 0.4);
  border-radius: 10px;
  padding: 10px 14px;
  margin-bottom: 16px;
  color: var(--text-secondary);
  background: rgba(79, 140, 255, 0.07);
}

.about-meta p {
  font-size: 0.88rem;
  color: var(--text-secondary);
}

.about-meta .version {
  margin-top: 6px;
  font-size: 0.8rem;
}

.about-actions {
  display: flex;
  justify-content: center;
  gap: 10px;
  margin-top: 14px;
}

.update-note {
  margin-top: 14px;
  padding: 12px;
  border-radius: 10px;
  background: var(--bg-secondary);
  border: 1px solid var(--border);
}

.update-note.available {
  border-color: var(--success);
}

.update-note.error {
  border-color: var(--danger);
  color: var(--danger);
}

@keyframes backdrop-in {
  from { opacity: 0; }
}

@keyframes panel-in {
  from { opacity: 0; transform: scale(0.94) translateY(10px); }
}

@media (prefers-reduced-motion: reduce) {
  .about-backdrop, .about-panel { animation: none; }
}
</style>
