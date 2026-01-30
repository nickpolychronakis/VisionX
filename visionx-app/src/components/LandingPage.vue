<script setup lang="ts">
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

const emit = defineEmits<{
  start: [];
}>();

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
    const info = await invoke<UpdateInfo>("check_for_updates");
    updateInfo.value = info;
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
</script>

<template>
  <div class="landing-page">
    <div class="hero">
      <img src="/logo.svg" alt="VisionX Logo" class="hero-logo" />
      <h2>Ανάλυση Βίντεο με Τεχνητή Νοημοσύνη</h2>
      <p class="tagline">
        Αυτόματη ανίχνευση και παρακολούθηση αντικειμένων σε βίντεο ασφαλείας
      </p>
    </div>

    <div class="content-scroll">
      <div class="dedication card">
        <p>
          Αφιερωμένο στα <strong>Γραφεία Προανακρίσεων</strong> των Τμημάτων Εξιχνιάσεων
          της Ελληνικής Αστυνομίας
        </p>
      </div>

      <div class="info-cards">
        <div class="features card">
          <h3>Πώς μπορεί να βοηθήσει</h3>
          <ul>
            <li>
              <span class="icon">🔍</span>
              <span><strong>Ανίχνευση Οχημάτων & Προσώπων</strong></span>
            </li>
            <li>
              <span class="icon">⏱️</span>
              <span><strong>Αυτόματη Χρονολόγηση</strong></span>
            </li>
            <li>
              <span class="icon">📊</span>
              <span><strong>Αναφορές HTML</strong></span>
            </li>
            <li>
              <span class="icon">🎯</span>
              <span><strong>Προσαρμοσμένη Αναζήτηση</strong></span>
            </li>
          </ul>
        </div>

        <div class="privacy card">
          <h3>🛡️ Απόρρητο</h3>
          <p><strong>100% Τοπική Εκτέλεση</strong></p>
          <p class="privacy-detail">
            Τα βίντεο δεν αποστέλλονται σε servers και δεν κοινοποιούνται σε τρίτους.
          </p>
        </div>
      </div>
    </div>

    <div class="actions">
      <button class="primary start-btn" @click="emit('start')">
        Έναρξη Ανάλυσης
      </button>
    </div>

    <footer class="credits">
      <p>
        Δημιουργήθηκε από <strong>Nick Polychronakis</strong>
      </p>
      <p class="contact">
        Επικοινωνία: <a href="mailto:nickpolychronakis@me.com">nickpolychronakis@me.com</a>
      </p>
      <div class="version-row">
        <span v-if="appVersion" class="version">Έκδοση {{ appVersion }}</span>
        <button
          class="update-check-btn"
          @click="checkForUpdates"
          :disabled="checkingUpdate"
        >
          {{ checkingUpdate ? 'Έλεγχος...' : 'Έλεγχος ενημερώσεων' }}
        </button>
      </div>

      <!-- Update notification -->
      <div v-if="updateInfo" class="update-notification" :class="{ available: updateInfo.available }">
        <template v-if="updateInfo.available">
          <p><strong>Διαθέσιμη νέα έκδοση: {{ updateInfo.version }}</strong></p>
          <div class="update-actions">
            <button
              class="primary"
              @click="installUpdate"
              :disabled="installingUpdate"
            >
              {{ installingUpdate ? 'Εγκατάσταση...' : (updateInfo.can_auto_update ? 'Εγκατάσταση' : 'Λήψη') }}
            </button>
            <button class="secondary" @click="dismissUpdate">Αργότερα</button>
          </div>
        </template>
        <template v-else>
          <p>Έχετε την τελευταία έκδοση!</p>
          <button class="secondary" @click="dismissUpdate">OK</button>
        </template>
      </div>

      <!-- Update error -->
      <div v-if="updateError" class="update-error">
        <p>{{ updateError }}</p>
        <button class="secondary" @click="updateError = ''">OK</button>
      </div>
    </footer>
  </div>
</template>

<style scoped>
.landing-page {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
}

.hero {
  text-align: center;
  padding: 10px 0;
  flex-shrink: 0;
}

.hero-logo {
  width: 80px;
  height: 80px;
  margin-bottom: 8px;
}

.hero h2 {
  font-size: 1.2rem;
  color: var(--text-primary);
  margin-bottom: 4px;
}

.tagline {
  color: var(--text-secondary);
  font-size: 0.9rem;
}

.content-scroll {
  flex: 1;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding: 10px 0;
}

.dedication {
  text-align: center;
  background: linear-gradient(135deg, var(--bg-card) 0%, rgba(233, 69, 96, 0.1) 100%);
  border: 1px solid var(--accent);
  padding: 14px;
  flex-shrink: 0;
}

.dedication p {
  font-size: 0.85rem;
  line-height: 1.5;
}

.info-cards {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}

.features h3,
.privacy h3 {
  margin-bottom: 12px;
  font-size: 0.9rem;
  color: var(--accent);
}

.features ul {
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.features li {
  display: flex;
  gap: 10px;
  align-items: center;
}

.features .icon {
  font-size: 1.1rem;
  flex-shrink: 0;
}

.features li strong {
  font-size: 0.85rem;
}

.features.card,
.privacy.card {
  padding: 14px;
}

.privacy p {
  font-size: 0.85rem;
  line-height: 1.4;
  margin-bottom: 4px;
}

.privacy-detail {
  color: var(--text-secondary);
  margin-bottom: 0;
}

.actions {
  display: flex;
  justify-content: center;
  padding: 12px 0;
  flex-shrink: 0;
}

.start-btn {
  padding: 14px 40px;
  font-size: 1rem;
}

.credits {
  text-align: center;
  padding-top: 12px;
  border-top: 1px solid var(--border);
  flex-shrink: 0;
}

.credits p {
  font-size: 0.8rem;
  color: var(--text-secondary);
  margin-bottom: 2px;
}

.contact a {
  color: var(--accent);
  text-decoration: none;
}

.contact a:hover {
  text-decoration: underline;
}

.version-row {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  margin-top: 4px;
}

.version {
  font-size: 0.75rem;
  color: var(--text-secondary);
  opacity: 0.7;
}

.update-check-btn {
  font-size: 0.7rem;
  padding: 4px 10px;
  background: transparent;
  color: var(--text-secondary);
  border: 1px solid var(--border);
}

.update-check-btn:hover:not(:disabled) {
  color: var(--accent);
  border-color: var(--accent);
}

.update-notification {
  margin-top: 12px;
  padding: 12px 16px;
  background: var(--bg-card);
  border-radius: 8px;
  border: 1px solid var(--border);
}

.update-notification.available {
  border-color: var(--success);
  background: rgba(78, 204, 163, 0.1);
}

.update-notification p {
  margin-bottom: 8px;
  color: var(--text-primary);
}

.update-actions {
  display: flex;
  gap: 8px;
  justify-content: center;
}

.update-actions button {
  font-size: 0.8rem;
  padding: 6px 16px;
}

.update-error {
  margin-top: 12px;
  padding: 12px 16px;
  background: rgba(233, 69, 96, 0.1);
  border: 1px solid var(--accent);
  border-radius: 8px;
}

.update-error p {
  margin-bottom: 8px;
  color: var(--accent);
  font-size: 0.85rem;
}
</style>
