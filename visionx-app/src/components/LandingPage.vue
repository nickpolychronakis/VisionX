<script setup lang="ts">
const emit = defineEmits<{
  start: [];
}>();
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

    <div class="dedication card">
      <p>
        Αφιερωμένο στα <strong>Γραφεία Προανακρίσεων</strong> των Τμημάτων Εξιχνιάσεων
        της Ελληνικής Αστυνομίας
      </p>
    </div>

    <div class="features card">
      <h3>Πώς μπορεί να βοηθήσει</h3>
      <ul>
        <li>
          <span class="icon">🔍</span>
          <div>
            <strong>Ανίχνευση Οχημάτων & Προσώπων</strong>
            <p>Αναγνώριση αυτοκινήτων, μοτοσικλετών και ατόμων σε βίντεο CCTV</p>
          </div>
        </li>
        <li>
          <span class="icon">⏱️</span>
          <div>
            <strong>Αυτόματη Χρονολόγηση</strong>
            <p>Καταγραφή ακριβούς χρόνου εμφάνισης κάθε αντικειμένου</p>
          </div>
        </li>
        <li>
          <span class="icon">📊</span>
          <div>
            <strong>Αναφορές HTML</strong>
            <p>Δημιουργία αναλυτικών αναφορών με screenshots και timestamps</p>
          </div>
        </li>
        <li>
          <span class="icon">🎯</span>
          <div>
            <strong>Προσαρμοσμένη Αναζήτηση</strong>
            <p>Αναζήτηση συγκεκριμένων αντικειμένων (π.χ. "λευκό αυτοκίνητο")</p>
          </div>
        </li>
      </ul>
    </div>

    <div class="privacy card">
      <h3>Απόρρητο & Ασφάλεια</h3>
      <div class="privacy-content">
        <span class="shield-icon">🛡️</span>
        <div>
          <p><strong>100% Τοπική Εκτέλεση</strong></p>
          <p>
            Η εφαρμογή εκτελείται αποκλειστικά στον υπολογιστή σας.
            Τα βίντεο και οι αναλύσεις <strong>δεν αποστέλλονται</strong> σε servers
            και <strong>δεν κοινοποιούνται</strong> σε τρίτους.
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
    </footer>
  </div>
</template>

<style scoped>
.landing-page {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.hero {
  text-align: center;
  padding: 20px 0;
}

.hero-logo {
  width: 120px;
  height: 120px;
  margin-bottom: 16px;
}

.hero h2 {
  font-size: 1.5rem;
  color: var(--text-primary);
  margin-bottom: 8px;
}

.tagline {
  color: var(--text-secondary);
  font-size: 1rem;
}

.dedication {
  text-align: center;
  background: linear-gradient(135deg, var(--bg-card) 0%, rgba(233, 69, 96, 0.1) 100%);
  border: 1px solid var(--accent);
}

.dedication p {
  font-size: 0.95rem;
  line-height: 1.6;
}

.features h3,
.privacy h3 {
  margin-bottom: 16px;
  font-size: 1rem;
  color: var(--accent);
}

.features ul {
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.features li {
  display: flex;
  gap: 14px;
  align-items: flex-start;
}

.features .icon {
  font-size: 1.5rem;
  flex-shrink: 0;
  width: 36px;
  text-align: center;
}

.features li strong {
  display: block;
  margin-bottom: 4px;
  font-size: 0.95rem;
}

.features li p {
  font-size: 0.85rem;
  color: var(--text-secondary);
  line-height: 1.4;
}

.privacy-content {
  display: flex;
  gap: 14px;
  align-items: flex-start;
}

.shield-icon {
  font-size: 2rem;
  flex-shrink: 0;
}

.privacy-content p {
  font-size: 0.9rem;
  line-height: 1.5;
  margin-bottom: 4px;
}

.privacy-content p:last-child {
  margin-bottom: 0;
  color: var(--text-secondary);
}

.actions {
  display: flex;
  justify-content: center;
  padding: 10px 0;
}

.start-btn {
  padding: 16px 48px;
  font-size: 1.1rem;
}

.credits {
  text-align: center;
  padding-top: 16px;
  border-top: 1px solid var(--border);
}

.credits p {
  font-size: 0.85rem;
  color: var(--text-secondary);
  margin-bottom: 4px;
}

.contact a {
  color: var(--accent);
  text-decoration: none;
}

.contact a:hover {
  text-decoration: underline;
}
</style>
