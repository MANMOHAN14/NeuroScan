/**
 * NeuroScan AI — Frontend Application Logic
 * ============================================
 * Handles form navigation, API communication, results
 * visualization with Chart.js, and UI interactions.
 */

const API_BASE = window.location.origin;

// ─── DOM Elements ───────────────────────────────────────
const assessmentForm = document.getElementById('assessment-form');
const steps = document.querySelectorAll('.form-step');
const progressSteps = document.querySelectorAll('.progress-step');
const progressFill = document.getElementById('progress-fill');
const prevBtn = document.getElementById('prev-btn');
const nextBtn = document.getElementById('next-btn');
const submitBtn = document.getElementById('submit-btn');
const resultsSection = document.getElementById('results');
const assessmentSection = document.getElementById('assessment');
const navbar = document.getElementById('navbar');

let currentStep = 1;
const totalSteps = 4;

// ─── Chart Instances (to destroy before re-creating) ────
let probabilityChart = null;
let radarChart = null;
let featureChart = null;

// ─── Initialize ─────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initFirebaseAuth();
    initSliders();
    initNavigation();
    initScrollEffects();
    loadModelInfo();
    loadFeatureImportance();
});

// ─── Firebase Authentication ────────────────────────────
function initFirebaseAuth() {
    const loginOverlay  = document.getElementById('login-overlay');
    const googleBtn     = document.getElementById('google-signin-btn');
    const navUser       = document.getElementById('nav-user');
    const navAvatar     = document.getElementById('nav-avatar');
    const navUsername   = document.getElementById('nav-username');
    const signoutBtn    = document.getElementById('signout-btn');

    // Wait for Firebase module to expose helpers (loaded via <script type="module">)
    const waitForFirebase = (tries = 0) => {
        if (window._fbOnAuth) {
            // Listen to auth state changes
            window._fbOnAuth((user) => {
                if (user) {
                    // User signed in — hide login overlay
                    loginOverlay.classList.add('hidden');
                    setTimeout(() => loginOverlay.style.display = 'none', 500);

                    // Show user in navbar
                    navAvatar.src  = user.photoURL || 'https://ui-avatars.com/api/?name=' + encodeURIComponent(user.displayName || 'User') + '&background=6366f1&color=fff';
                    navUsername.textContent = (user.displayName || user.email || '').split(' ')[0];
                    navUser.style.display = 'flex';
                } else {
                    // Not signed in — show login overlay
                    loginOverlay.style.display = 'flex';
                    loginOverlay.classList.remove('hidden');
                    navUser.style.display = 'none';
                }
            });

            // Google Sign-In button
            googleBtn.addEventListener('click', async () => {
                googleBtn.classList.add('loading');
                googleBtn.textContent = 'Signing in...';

                // --- DEMO BYPASS: FAKE LOGIN IF NO FIREBASE CONFIG ---
                if (!window._isFirebaseConfigured) {
                    setTimeout(() => {
                        googleBtn.classList.remove('loading');
                        googleBtn.innerHTML = `<svg width="22" height="22" viewBox="0 0 48 48"><path fill="#FFC107" d="M43.611 20.083H42V20H24v8h11.303c-1.649 4.657-6.08 8-11.303 8-6.627 0-12-5.373-12-12s5.373-12 12-12c3.059 0 5.842 1.154 7.961 3.039l5.657-5.657C34.046 6.053 29.268 4 24 4 12.955 4 4 12.955 4 24s8.955 20 20 20 20-8.955 20-20c0-1.341-.138-2.65-.389-3.917z"/><path fill="#FF3D00" d="m6.306 14.691 6.571 4.819C14.655 15.108 18.961 12 24 12c3.059 0 5.842 1.154 7.961 3.039l5.657-5.657C34.046 6.053 29.268 4 24 4 16.318 4 9.656 8.337 6.306 14.691z"/><path fill="#4CAF50" d="M24 44c5.166 0 9.86-1.977 13.409-5.192l-6.19-5.238A11.91 11.91 0 0 1 24 36c-5.202 0-9.619-3.317-11.283-7.946l-6.522 5.025C9.505 39.556 16.227 44 24 44z"/><path fill="#1976D2" d="M43.611 20.083H42V20H24v8h11.303a12.04 12.04 0 0 1-4.087 5.571l.003-.002 6.19 5.238C36.971 39.205 44 34 44 24c0-1.341-.138-2.65-.389-3.917z"/></svg> Continue with Google`;
                        
                        loginOverlay.classList.add('hidden');
                        setTimeout(() => loginOverlay.style.display = 'none', 500);

                        navAvatar.src = 'https://ui-avatars.com/api/?name=Demo+User&background=6366f1&color=fff';
                        navUsername.textContent = 'Demo User';
                        navUser.style.display = 'flex';
                    }, 1200);
                    return;
                }
                
                try {
                    await window._fbSignIn();
                } catch (err) {
                    googleBtn.classList.remove('loading');
                    googleBtn.innerHTML = `<svg width="22" height="22" viewBox="0 0 48 48"><path fill="#FFC107" d="M43.611 20.083H42V20H24v8h11.303c-1.649 4.657-6.08 8-11.303 8-6.627 0-12-5.373-12-12s5.373-12 12-12c3.059 0 5.842 1.154 7.961 3.039l5.657-5.657C34.046 6.053 29.268 4 24 4 12.955 4 4 12.955 4 24s8.955 20 20 20 20-8.955 20-20c0-1.341-.138-2.65-.389-3.917z"/><path fill="#FF3D00" d="m6.306 14.691 6.571 4.819C14.655 15.108 18.961 12 24 12c3.059 0 5.842 1.154 7.961 3.039l5.657-5.657C34.046 6.053 29.268 4 24 4 16.318 4 9.656 8.337 6.306 14.691z"/><path fill="#4CAF50" d="M24 44c5.166 0 9.86-1.977 13.409-5.192l-6.19-5.238A11.91 11.91 0 0 1 24 36c-5.202 0-9.619-3.317-11.283-7.946l-6.522 5.025C9.505 39.556 16.227 44 24 44z"/><path fill="#1976D2" d="M43.611 20.083H42V20H24v8h11.303a12.04 12.04 0 0 1-4.087 5.571l.003-.002 6.19 5.238C36.971 39.205 44 34 44 24c0-1.341-.138-2.65-.389-3.917z"/></svg> Continue with Google`;
                    if (err.code !== 'auth/popup-closed-by-user') {
                        alert('Sign-in failed: ' + err.message);
                    }
                }
            });

            // Sign-Out button
            signoutBtn.addEventListener('click', async () => {
                if (!window._isFirebaseConfigured) {
                    loginOverlay.style.display = 'flex';
                    loginOverlay.classList.remove('hidden');
                    navUser.style.display = 'none';
                    return;
                }
                await window._fbSignOut();
            });

        } else if (tries < 30) {
            // Firebase module not ready yet — retry every 200ms (up to 6s)
            setTimeout(() => waitForFirebase(tries + 1), 200);
        } else {
            // Firebase config not set — show demo bypass button
            console.warn('Firebase not configured. Showing demo bypass.');
            const demoNote = document.createElement('button');
            demoNote.className = 'btn-google';
            demoNote.style.marginTop = '10px';
            demoNote.style.background = 'rgba(99,102,241,0.15)';
            demoNote.style.color = '#a5b4fc';
            demoNote.style.border = '1px solid rgba(99,102,241,0.3)';
            demoNote.textContent = '🎓 Continue as Demo (Firebase not configured)';
            demoNote.addEventListener('click', () => {
                loginOverlay.classList.add('hidden');
                setTimeout(() => loginOverlay.style.display = 'none', 500);
                navUsername.textContent = 'Demo User';
                navUser.style.display = 'flex';
            });
            document.querySelector('.login-card').appendChild(demoNote);
        }
    };
    waitForFirebase();
}



// ─── Slider Value Display ───────────────────────────────
function initSliders() {
    const sliders = document.querySelectorAll('.slider');
    sliders.forEach(slider => {
        const valueDisplay = document.getElementById(slider.id + '_val');
        if (valueDisplay) {
            valueDisplay.textContent = slider.value;
            slider.addEventListener('input', () => {
                valueDisplay.textContent = slider.value;
                updateSliderTrack(slider);
            });
            updateSliderTrack(slider);
        }
    });
}

function updateSliderTrack(slider) {
    const min = parseFloat(slider.min);
    const max = parseFloat(slider.max);
    const val = parseFloat(slider.value);
    const percent = ((val - min) / (max - min)) * 100;
    slider.style.background = `linear-gradient(to right, 
        #6366f1 0%, #06b6d4 ${percent}%, 
        rgba(255,255,255,0.08) ${percent}%)`;
}

// ─── Form Navigation ────────────────────────────────────
function initNavigation() {
    prevBtn.addEventListener('click', () => navigateStep(-1));
    nextBtn.addEventListener('click', () => navigateStep(1));
    assessmentForm.addEventListener('submit', handleSubmit);

    // New Assessment button
    document.getElementById('new-assessment-btn').addEventListener('click', () => {
        resultsSection.style.display = 'none';
        assessmentSection.style.display = 'block';
        currentStep = 1;
        updateStepUI();
        assessmentForm.reset();
        initSliders();
        window.scrollTo({
            top: assessmentSection.offsetTop - 80,
            behavior: 'smooth'
        });
    });

    // Mobile hamburger
    const hamburger = document.getElementById('nav-hamburger');
    const navLinks = document.querySelector('.nav-links');
    hamburger.addEventListener('click', () => {
        navLinks.classList.toggle('open');
    });

    // Nav link clicks on mobile
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', () => {
            navLinks.classList.remove('open');
        });
    });
}

function navigateStep(direction) {
    const newStep = currentStep + direction;
    if (newStep < 1 || newStep > totalSteps) return;
    currentStep = newStep;
    updateStepUI();
}

function updateStepUI() {
    // Show/hide steps
    steps.forEach(step => step.classList.remove('active'));
    document.getElementById(`step-${currentStep}`).classList.add('active');

    // Update progress
    const progressPercent = (currentStep / totalSteps) * 100;
    progressFill.style.width = `${progressPercent}%`;

    // Update step indicators
    progressSteps.forEach((step, index) => {
        step.classList.remove('active', 'completed');
        if (index + 1 === currentStep) {
            step.classList.add('active');
        } else if (index + 1 < currentStep) {
            step.classList.add('completed');
        }
    });

    // Show/hide buttons
    prevBtn.disabled = currentStep === 1;
    prevBtn.style.visibility = currentStep === 1 ? 'hidden' : 'visible';

    if (currentStep === totalSteps) {
        nextBtn.style.display = 'none';
        submitBtn.style.display = 'inline-flex';
    } else {
        nextBtn.style.display = 'inline-flex';
        submitBtn.style.display = 'none';
    }
}

// ─── Scroll Effects ─────────────────────────────────────
function initScrollEffects() {
    // Navbar scroll effect
    window.addEventListener('scroll', () => {
        if (window.scrollY > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }

        // Active nav link
        updateActiveNavLink();
    });

    // Intersection Observer for fade-in animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);

    // Observe cards
    document.querySelectorAll('.info-card, .stats-card, .about-card').forEach(card => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        card.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(card);
    });
}

function updateActiveNavLink() {
    const sections = ['hero', 'assessment', 'model-stats', 'about'];
    const scrollPos = window.scrollY + 150;

    sections.forEach(id => {
        const section = document.getElementById(id);
        if (!section) return;
        const link = document.querySelector(`.nav-link[href="#${id}"]`);
        if (!link) return;

        if (scrollPos >= section.offsetTop && 
            scrollPos < section.offsetTop + section.offsetHeight) {
            document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
            link.classList.add('active');
        }
    });
}

// ─── Load Model Info ────────────────────────────────────
async function loadModelInfo() {
    try {
        const response = await fetch(`${API_BASE}/api/model-info`);
        if (!response.ok) throw new Error('API unavailable');
        const data = await response.json();

        // Hero stat
        const accElem = document.getElementById('stat-accuracy');
        if (accElem) {
            animateCounter(accElem, 0, (data.accuracy * 100), '%', 1500);
        }

        // Stats section
        document.getElementById('stats-accuracy').textContent = 
            (data.accuracy * 100).toFixed(1) + '%';
        document.getElementById('stats-f1').textContent = 
            (data.f1_score * 100).toFixed(1) + '%';
        document.getElementById('stats-cv').textContent = 
            (data.cv_mean * 100).toFixed(1) + '%';
        document.getElementById('stats-model').textContent = data.model_name;
    } catch (err) {
        console.log('Model info not available:', err.message);
        document.getElementById('stat-accuracy').textContent = '95%';
    }
}

// ─── Load Feature Importance ────────────────────────────
async function loadFeatureImportance() {
    try {
        const response = await fetch(`${API_BASE}/api/feature-importance`);
        if (!response.ok) throw new Error('API unavailable');
        const data = await response.json();

        const features = Object.keys(data);
        const values = Object.values(data);

        const ctx = document.getElementById('feature-importance-chart');
        if (!ctx) return;

        if (featureChart) featureChart.destroy();

        featureChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: features.map(f => f.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())),
                datasets: [{
                    label: 'Importance',
                    data: values.map(v => (v * 100).toFixed(2)),
                    backgroundColor: generateGradientColors(features.length),
                    borderRadius: 6,
                    borderSkipped: false,
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: 'rgba(15, 15, 46, 0.9)',
                        titleColor: '#e8e8f0',
                        bodyColor: '#9495b7',
                        borderColor: 'rgba(255,255,255,0.1)',
                        borderWidth: 1,
                        cornerRadius: 8,
                        padding: 12,
                        callbacks: {
                            label: (ctx) => `Importance: ${ctx.raw}%`
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255,255,255,0.04)' },
                        ticks: { color: '#6b6d8a', font: { size: 11 } },
                        title: { display: true, text: 'Importance (%)', color: '#9495b7' }
                    },
                    y: {
                        grid: { display: false },
                        ticks: { color: '#9495b7', font: { size: 11 } }
                    }
                }
            }
        });

        // Set canvas height based on features count
        ctx.parentElement.style.height = `${Math.max(400, features.length * 32)}px`;

    } catch (err) {
        console.log('Feature importance not available:', err.message);
    }
}

// ─── Form Submission ────────────────────────────────────
async function handleSubmit(e) {
    e.preventDefault();

    const btnText = submitBtn.querySelector('.btn-text');
    const btnLoading = submitBtn.querySelector('.btn-loading');
    const btnIcon = submitBtn.querySelector('.btn-icon');

    // Show loading
    btnText.style.display = 'none';
    btnIcon.style.display = 'none';
    btnLoading.style.display = 'flex';
    submitBtn.disabled = true;

    // Collect form data
    const formData = collectFormData();

    try {
        const response = await fetch(`${API_BASE}/api/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.error || 'Prediction failed');
        }

        const result = await response.json();
        displayResults(result, formData);

    } catch (err) {
        alert('Error: ' + err.message + '\n\nMake sure the backend server is running on port 5000.');
    } finally {
        btnText.style.display = 'flex';
        btnIcon.style.display = 'block';
        btnLoading.style.display = 'none';
        submitBtn.disabled = false;
    }
}

function collectFormData() {
    return {
        age: parseInt(document.getElementById('age').value),
        gender: parseInt(document.getElementById('gender').value),
        reading_score: parseFloat(document.getElementById('reading_score').value),
        writing_score: parseFloat(document.getElementById('writing_score').value),
        math_score: parseFloat(document.getElementById('math_score').value),
        attention_span: parseInt(document.getElementById('attention_span').value),
        memory_score: parseFloat(document.getElementById('memory_score').value),
        processing_speed: parseFloat(document.getElementById('processing_speed').value),
        hyperactivity_score: parseInt(document.getElementById('hyperactivity_score').value),
        impulsivity_score: parseInt(document.getElementById('impulsivity_score').value),
        social_skills: parseInt(document.getElementById('social_skills').value),
        fine_motor_skills: parseFloat(document.getElementById('fine_motor_skills').value),
        phonological_awareness: parseFloat(document.getElementById('phonological_awareness').value),
        vocabulary_score: parseFloat(document.getElementById('vocabulary_score').value),
        parent_education: document.getElementById('parent_education').value,
        family_history_ld: parseInt(document.getElementById('family_history_ld').value),
        sleep_hours: parseFloat(document.getElementById('sleep_hours').value),
        screen_time: parseFloat(document.getElementById('screen_time').value),
        class_participation: parseInt(document.getElementById('class_participation').value),
        emotional_stability: parseInt(document.getElementById('emotional_stability').value),
    };
}

// ─── Display Results ────────────────────────────────────
function displayResults(result, formData) {
    // Hide assessment, show results
    assessmentSection.style.display = 'none';
    resultsSection.style.display = 'block';

    // Scroll to results
    window.scrollTo({
        top: resultsSection.offsetTop - 80,
        behavior: 'smooth'
    });

    // Icon mapping
    const icons = {
        'No Disability': '✅',
        'Dyslexia': '📖',
        'Dysgraphia': '✍️',
        'Dyscalculia': '🔢',
        'ADHD': '⚡'
    };

    // Set main result
    document.getElementById('result-icon').textContent = icons[result.prediction] || '🧠';
    document.getElementById('result-label').textContent = result.prediction;
    document.getElementById('result-label').style.color = result.color;
    document.getElementById('confidence-percent').textContent = Math.round(result.confidence);
    document.getElementById('result-description').innerHTML = `<p>${result.description}</p>`;

    // Confidence ring animation
    const circle = document.getElementById('confidence-circle');
    const circumference = 2 * Math.PI * 52;
    const offset = circumference - (result.confidence / 100) * circumference;
    circle.style.color = result.color;
    setTimeout(() => {
        circle.style.strokeDashoffset = offset;
    }, 100);

    // Risk badge
    const riskBadge = document.getElementById('risk-badge');
    riskBadge.textContent = result.risk_level;
    riskBadge.className = 'risk-badge risk-' + result.risk_level.toLowerCase();

    // Recommendations
    const recList = document.getElementById('recommendations-list');
    recList.innerHTML = '';
    result.recommendations.forEach(rec => {
        const li = document.createElement('li');
        li.textContent = rec;
        recList.appendChild(li);
    });

    // Charts
    createProbabilityChart(result.class_probabilities, result.color);
    createRadarChart(formData);
}

// ─── Probability Chart ──────────────────────────────────
function createProbabilityChart(probabilities, highlightColor) {
    const ctx = document.getElementById('probability-chart');
    if (probabilityChart) probabilityChart.destroy();

    const labels = Object.keys(probabilities);
    const values = Object.values(probabilities);

    const colors = [
        '#10b981', // No Disability - green
        '#f59e0b', // Dyslexia - amber
        '#8b5cf6', // Dysgraphia - purple
        '#ec4899', // Dyscalculia - pink
        '#ef4444', // ADHD - red
    ];

    probabilityChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: colors.map(c => c + '33'),
                borderColor: colors,
                borderWidth: 2,
                hoverBackgroundColor: colors.map(c => c + '66'),
                hoverBorderWidth: 3,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            cutout: '55%',
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: '#9495b7',
                        padding: 16,
                        usePointStyle: true,
                        pointStyleWidth: 10,
                        font: { size: 11 }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(15, 15, 46, 0.95)',
                    titleColor: '#e8e8f0',
                    bodyColor: '#9495b7',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1,
                    cornerRadius: 8,
                    padding: 12,
                    callbacks: {
                        label: (ctx) => `${ctx.label}: ${ctx.raw}%`
                    }
                }
            }
        }
    });
}

// ─── Radar Chart ────────────────────────────────────────
function createRadarChart(formData) {
    const ctx = document.getElementById('radar-chart');
    if (radarChart) radarChart.destroy();

    const radarData = {
        labels: ['Reading', 'Writing', 'Math', 'Attention', 'Memory', 'Motor Skills', 'Social', 'Emotional'],
        datasets: [{
            label: 'Student Profile',
            data: [
                formData.reading_score,
                formData.writing_score,
                formData.math_score,
                formData.attention_span * 10,
                formData.memory_score,
                formData.fine_motor_skills,
                formData.social_skills * 10,
                formData.emotional_stability * 10,
            ],
            backgroundColor: 'rgba(99, 102, 241, 0.15)',
            borderColor: '#6366f1',
            borderWidth: 2,
            pointBackgroundColor: '#6366f1',
            pointBorderColor: '#fff',
            pointBorderWidth: 1,
            pointRadius: 4,
            pointHoverRadius: 6,
        }]
    };

    radarChart = new Chart(ctx, {
        type: 'radar',
        data: radarData,
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    labels: {
                        color: '#9495b7',
                        font: { size: 12 }
                    }
                }
            },
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)',
                    },
                    angleLines: {
                        color: 'rgba(255, 255, 255, 0.05)',
                    },
                    pointLabels: {
                        color: '#9495b7',
                        font: { size: 11 }
                    },
                    ticks: {
                        color: '#6b6d8a',
                        backdropColor: 'transparent',
                        font: { size: 9 },
                        stepSize: 25
                    }
                }
            }
        }
    });
}

// ─── Utility: Animated Counter ──────────────────────────
function animateCounter(element, start, end, suffix = '', duration = 1500) {
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Ease out quad
        const eased = 1 - (1 - progress) * (1 - progress);
        const current = Math.round(start + (end - start) * eased);

        element.textContent = current + suffix;

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}

// ─── Utility: Generate Gradient Colors ──────────────────
function generateGradientColors(count) {
    const colors = [];
    for (let i = 0; i < count; i++) {
        const hue = 250 + (i * (140 / count));
        colors.push(`hsla(${hue % 360}, 70%, 60%, 0.75)`);
    }
    return colors;
}
