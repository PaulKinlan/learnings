/**
 * Image Decision Lab - Interactive testing harness for multimodal System One models
 * Supports: drag-and-drop upload, clipboard paste, preset visual test images,
 * JSON question editing, client-side vision decider, local loopback decider,
 * and JevImageBench v0.1.5 explorer table.
 */

import { gate, loopbackEndpoint, validateAnswers } from './core.js';
import {
  BENCHMARK_META,
  JEV_IMAGE_BENCH_DATA,
  filterImageBench,
  sortImageBench
} from './image-bench-data.js';

export const $ = id => typeof document !== 'undefined' ? document.getElementById(id) : null;

export function el(tag, text, cls) {
  if (typeof document === 'undefined') {
    return {
      tagName: tag.toUpperCase(),
      textContent: text || '',
      className: cls || '',
      style: {},
      children: [],
      append(...items) {
        for (const item of items) {
          if (typeof item === 'string') {
            this.textContent += item;
          } else {
            this.children.push(item);
            if (item.textContent) {
              this.textContent += item.textContent;
            }
          }
        }
      },
      setAttribute(k, v) { this[k] = v; }
    };
  }
  const n = document.createElement(tag);
  if (text !== undefined) n.textContent = text;
  if (cls) n.className = cls;
  return n;
}

// 4 Canvas-rendered preset images
export function drawUIPreset(canvas) {
  canvas.width = 600;
  canvas.height = 380;
  const ctx = canvas.getContext('2d');

  // Background page
  ctx.fillStyle = '#f1f5f9';
  ctx.fillRect(0, 0, 600, 380);

  // App header
  ctx.fillStyle = '#1e293b';
  ctx.fillRect(0, 0, 600, 50);
  ctx.fillStyle = '#ffffff';
  ctx.font = 'bold 16px sans-serif';
  ctx.fillText('⚡ QuantumCart Store · Checkout', 20, 32);

  // Cart summary behind modal
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(30, 70, 540, 280);
  ctx.fillStyle = '#64748b';
  ctx.font = '14px sans-serif';
  ctx.fillText('Order Summary: 1 item ($149.00) + Shipping ($15.00) = Total: $164.00', 50, 110);
  ctx.fillText('Shipping to: 450 Mission St, San Francisco CA', 50, 135);

  // Dimmed overlay
  ctx.fillStyle = 'rgba(15, 23, 42, 0.65)';
  ctx.fillRect(0, 0, 600, 380);

  // Error Modal Dialog
  const mx = 90, my = 75, mw = 420, mh = 230;
  ctx.fillStyle = '#ffffff';
  ctx.shadowColor = 'rgba(0, 0, 0, 0.35)';
  ctx.shadowBlur = 20;
  ctx.shadowOffsetY = 8;
  ctx.beginPath();
  ctx.roundRect(mx, my, mw, mh, 10);
  ctx.fill();
  ctx.shadowColor = 'transparent';

  // Error icon & title
  ctx.fillStyle = '#ef4444';
  ctx.beginPath();
  ctx.arc(mx + 36, my + 42, 16, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = '#ffffff';
  ctx.font = 'bold 18px sans-serif';
  ctx.fillText('!', mx + 33, my + 48);

  ctx.fillStyle = '#0f172a';
  ctx.font = 'bold 18px sans-serif';
  ctx.fillText('Payment Authorization Failed', mx + 62, my + 48);

  // Error message
  ctx.fillStyle = '#475569';
  ctx.font = '14px sans-serif';
  ctx.fillText('Your financial institution declined authorization code #402.', mx + 30, my + 88);
  ctx.fillText('Reason: Insufficient funds or velocity check exceeded.', mx + 30, my + 110);
  ctx.fillText('Please select an action to proceed:', mx + 30, my + 134);

  // Action Button 1: Retry Payment (Primary)
  ctx.fillStyle = '#0284c7';
  ctx.beginPath();
  ctx.roundRect(mx + 30, my + 160, 160, 42, 6);
  ctx.fill();
  ctx.fillStyle = '#ffffff';
  ctx.font = 'bold 14px sans-serif';
  ctx.fillText('Retry Payment', mx + 62, my + 186);

  // Action Button 2: Cancel Order (Secondary)
  ctx.fillStyle = '#e2e8f0';
  ctx.beginPath();
  ctx.roundRect(mx + 210, my + 160, 160, 42, 6);
  ctx.fill();
  ctx.fillStyle = '#334155';
  ctx.font = 'bold 14px sans-serif';
  ctx.fillText('Cancel Order', mx + 248, my + 186);

  return canvas.toDataURL('image/png');
}

export function drawInvoicePreset(canvas) {
  canvas.width = 540;
  canvas.height = 420;
  const ctx = canvas.getContext('2d');

  // Paper sheet background
  ctx.fillStyle = '#e2e8f0';
  ctx.fillRect(0, 0, 540, 420);
  ctx.fillStyle = '#ffffff';
  ctx.shadowColor = 'rgba(0, 0, 0, 0.15)';
  ctx.shadowBlur = 12;
  ctx.shadowOffsetY = 4;
  ctx.fillRect(30, 20, 480, 380);
  ctx.shadowColor = 'transparent';

  // Invoice header
  ctx.fillStyle = '#0f172a';
  ctx.font = 'bold 20px monospace';
  ctx.fillText('ACME CLOUD COMPUTE INC.', 50, 58);

  ctx.fillStyle = '#64748b';
  ctx.font = '12px monospace';
  ctx.fillText('INVOICE #: INV-2026-9042', 50, 80);
  ctx.fillText('DATE: 2026-10-02  |  TERMS: NET 30', 50, 98);
  ctx.fillText('BILL TO: Paul Kinlan Development Labs', 50, 116);

  // Line separator
  ctx.strokeStyle = '#cbd5e1';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(50, 130);
  ctx.lineTo(490, 130);
  ctx.stroke();

  // Table items
  ctx.fillStyle = '#1e293b';
  ctx.font = '13px monospace';
  ctx.fillText('DESCRIPTION                          QTY    AMOUNT', 50, 155);

  ctx.strokeStyle = '#e2e8f0';
  ctx.beginPath();
  ctx.moveTo(50, 165);
  ctx.lineTo(490, 165);
  ctx.stroke();

  ctx.fillStyle = '#334155';
  ctx.fillText('1. H100 Cluster GPU Compute (60 hrs)     1   $1,200.00', 50, 192);
  ctx.fillText('2. High-Bandwidth S3 Storage (12 TB)     1     $180.00', 50, 218);
  ctx.fillText('3. Multi-Region Network Egress           1      $40.50', 50, 244);

  ctx.beginPath();
  ctx.moveTo(50, 270);
  ctx.lineTo(490, 270);
  ctx.stroke();

  // Subtotal and Total
  ctx.fillStyle = '#0f172a';
  ctx.font = 'bold 15px monospace';
  ctx.fillText('SUBTOTAL:                          $1,420.50', 160, 298);
  ctx.fillText('TAX (0.0%):                            $0.00', 160, 320);
  ctx.fillStyle = '#047857';
  ctx.font = 'bold 17px monospace';
  ctx.fillText('TOTAL DUE:                         $1,420.50', 160, 348);

  // Legibility Stamp
  ctx.strokeStyle = '#059669';
  ctx.lineWidth = 2;
  ctx.strokeRect(50, 305, 95, 45);
  ctx.fillStyle = '#059669';
  ctx.font = 'bold 13px sans-serif';
  ctx.fillText('PAID IN FULL', 56, 332);

  return canvas.toDataURL('image/png');
}

export function drawSecurityPreset(canvas) {
  canvas.width = 620;
  canvas.height = 360;
  const ctx = canvas.getContext('2d');

  // Dark IDE window
  ctx.fillStyle = '#0f172a';
  ctx.fillRect(0, 0, 620, 360);

  // Window title bar
  ctx.fillStyle = '#1e293b';
  ctx.fillRect(0, 0, 620, 36);

  // Mac buttons
  const colors = ['#ef4444', '#f59e0b', '#10b981'];
  colors.forEach((c, i) => {
    ctx.fillStyle = c;
    ctx.beginPath();
    ctx.arc(20 + i * 18, 18, 5, 0, Math.PI * 2);
    ctx.fill();
  });

  ctx.fillStyle = '#94a3b8';
  ctx.font = '12px monospace';
  ctx.fillText('src/config/secrets.py — confidential repository', 90, 22);

  // Line numbers gutter
  ctx.fillStyle = '#334155';
  ctx.fillRect(0, 36, 45, 324);
  ctx.fillStyle = '#64748b';
  ctx.font = '12px monospace';
  for (let i = 1; i <= 9; i++) {
    ctx.fillText(String(i), 18, 58 + (i - 1) * 28);
  }

  // Code lines
  ctx.font = '13px monospace';
  ctx.fillStyle = '#64748b';
  ctx.fillText('# PRODUCTION CONFIGURATION — DO NOT COMMIT CREDENTIALS', 60, 58);

  ctx.fillStyle = '#38bdf8';
  ctx.fillText('import', 60, 86);
  ctx.fillStyle = '#f8fafc';
  ctx.fillText(' os', 115, 86);

  ctx.fillStyle = '#cbd5e1';
  ctx.fillText('ENVIRONMENT = ', 60, 114);
  ctx.fillStyle = '#4ade80';
  ctx.fillText('"production"', 165, 114);

  // Exposed API Key Line (Highlighted)
  ctx.fillStyle = 'rgba(239, 68, 68, 0.2)';
  ctx.fillRect(45, 126, 575, 32);

  ctx.fillStyle = '#cbd5e1';
  ctx.fillText('OPENAI_API_KEY = ', 60, 146);
  ctx.fillStyle = '#f87171';
  ctx.font = 'bold 13px monospace';
  ctx.fillText('"sk-proj-91aB4cE9829f0a82b99214cd91807aa3bc"', 188, 146);

  // Another secret
  ctx.fillStyle = 'rgba(239, 68, 68, 0.2)';
  ctx.fillRect(45, 154, 575, 32);

  ctx.font = '13px monospace';
  ctx.fillStyle = '#cbd5e1';
  ctx.fillText('DATABASE_DSN = ', 60, 174);
  ctx.fillStyle = '#f87171';
  ctx.fillText('"postgres://admin:MasterPass2026@10.0.1.4:5432/core"', 172, 174);

  ctx.fillStyle = '#cbd5e1';
  ctx.fillText('DEBUG = ', 60, 202);
  ctx.fillStyle = '#fbbf24';
  ctx.fillText('False', 120, 202);

  // Red alert box at bottom
  ctx.fillStyle = 'rgba(239, 68, 68, 0.9)';
  ctx.beginPath();
  ctx.roundRect(60, 250, 480, 50, 6);
  ctx.fill();
  ctx.fillStyle = '#ffffff';
  ctx.font = 'bold 14px sans-serif';
  ctx.fillText('⚠️ DETECTED LIVE CREDENTIALS IN ACTIVE REPOSITORY BUFFER', 75, 280);

  return canvas.toDataURL('image/png');
}

export function drawNavigationPreset(canvas) {
  canvas.width = 600;
  canvas.height = 380;
  const ctx = canvas.getContext('2d');

  // Indoor hallway view (perspective)
  // Ceiling
  ctx.fillStyle = '#1e293b';
  ctx.fillRect(0, 0, 600, 150);

  // Floor
  ctx.fillStyle = '#334155';
  ctx.fillRect(0, 150, 600, 230);

  // Floor perspective lines
  ctx.strokeStyle = '#475569';
  ctx.lineWidth = 2;
  for (let i = 0; i <= 600; i += 75) {
    ctx.beginPath();
    ctx.moveTo(i, 380);
    ctx.lineTo(300, 160);
    ctx.stroke();
  }

  // Left corridor (Open / Green)
  ctx.fillStyle = '#064e3b';
  ctx.beginPath();
  ctx.moveTo(0, 120);
  ctx.lineTo(210, 150);
  ctx.lineTo(210, 290);
  ctx.lineTo(0, 380);
  ctx.fill();

  ctx.fillStyle = '#10b981';
  ctx.font = 'bold 18px sans-serif';
  ctx.fillText('← CLEAR CORRIDOR (3.2m)', 25, 230);

  // Right barrier / wall (Yellow / Hazard)
  ctx.fillStyle = '#713f12';
  ctx.beginPath();
  ctx.moveTo(390, 150);
  ctx.lineTo(600, 120);
  ctx.lineTo(600, 380);
  ctx.lineTo(390, 290);
  ctx.fill();

  ctx.fillStyle = '#fbbf24';
  ctx.font = 'bold 16px sans-serif';
  ctx.fillText('WALL / BLOCKED →', 420, 230);

  // Center obstacle: Red collision pylon
  ctx.fillStyle = '#dc2626';
  ctx.beginPath();
  ctx.moveTo(270, 280);
  ctx.lineTo(290, 175);
  ctx.lineTo(310, 175);
  ctx.lineTo(330, 280);
  ctx.closePath();
  ctx.fill();

  // White hazard stripes on pylon
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(278, 240, 44, 12);
  ctx.fillRect(285, 205, 30, 10);

  // Warning text above obstacle
  ctx.fillStyle = '#ef4444';
  ctx.font = 'bold 16px sans-serif';
  ctx.fillText('⚠️ OBSTACLE AHEAD: 0.9m', 200, 135);

  // Robot HUD overlay
  ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
  ctx.fillRect(0, 330, 600, 50);
  ctx.fillStyle = '#38bdf8';
  ctx.font = 'bold 13px monospace';
  ctx.fillText('FORWARD VELOCITY: 1.2 m/s | LIDAR: OBSTRUCTED | SAFE VECTOR: TURN LEFT', 20, 360);

  return canvas.toDataURL('image/png');
}

export const PRESETS = {
  ui: {
    id: 'ui',
    name: 'UI Error Modal',
    description: 'Web application checkout error modal: determine if user is blocked and which button to click.',
    draw: drawUIPreset,
    spec: {
      title: 'UI Error Modal Triage',
      description: 'Examine web application screenshot containing an error modal to determine blocker status and action.',
      questions: {
        is_blocked: {
          type: 'noul',
          instructions: 'Is the user currently blocked from completing checkout by an error modal?',
          criteria: {
            true: 'An active error dialog obstructs completion',
            false: 'User can proceed normally without obstruction'
          }
        },
        action: {
          type: 'choice',
          instructions: 'Which button action should automated recovery or user guidance recommend?',
          criteria: {
            retry: 'Click Retry Payment to attempt card re-authorization',
            cancel: 'Click Cancel Order to abandon the checkout session',
            dismiss: 'Close modal and edit billing address details',
            unclear: 'Screenshot does not present a legible recovery button'
          }
        },
        severity: {
          type: 'score',
          instructions: 'Rate the operational impact of this modal on the checkout funnel.',
          criteria: [
            'Cosmetic notice (no user impact)',
            'Warning (workaround available)',
            'Hard blocking transactional failure'
          ]
        }
      }
    },
    answers: {
      is_blocked: {
        type: 'noul',
        noul: 0.965,
        confidence: 0.965
      },
      action: {
        type: 'choice',
        choice: 'retry',
        confidence: 0.912,
        probabilities: {
          retry: 0.885,
          cancel: 0.080,
          dismiss: 0.025,
          unclear: 0.010
        }
      },
      severity: {
        type: 'score',
        score: 1.94,
        confidence: 0.940,
        probabilities: {
          '0': 0.010,
          '1': 0.040,
          '2': 0.950
        }
      }
    }
  },
  invoice: {
    id: 'invoice',
    name: 'Invoice / Receipt',
    description: 'Cloud hosting invoice: verify text legibility, financial category, and expense approval tier.',
    draw: drawInvoicePreset,
    spec: {
      title: 'Invoice Legibility & Expense Categorization',
      description: 'Inspect an uploaded document image to verify readability and route to the correct accounting category.',
      questions: {
        is_legible: {
          type: 'noul',
          instructions: 'Is the text, invoice total, and vendor name clearly legible for automated accounting?',
          criteria: {
            true: 'All critical financial figures and vendor names are sharp and readable',
            false: 'Image is blurry, cropped, occluded or unreadable'
          }
        },
        expense_category: {
          type: 'choice',
          instructions: 'Which financial ledger expense category applies to this bill?',
          criteria: {
            cloud_hosting: 'Cloud infrastructure, server compute and hosting services',
            travel_meals: 'Airfare, lodging, meals and business entertainment',
            office_supplies: 'Physical office hardware, stationery and supplies',
            unclear: 'Vendor or line items cannot be matched to known categories'
          }
        },
        approval_tier: {
          type: 'score',
          instructions: 'Determine the financial approval tier based on the total invoice amount.',
          criteria: [
            'Tier 1: Under $100 (auto-approve)',
            'Tier 2: $100 – $1,000 (manager sign-off)',
            'Tier 3: Over $1,000 (executive approval required)'
          ]
        }
      }
    },
    answers: {
      is_legible: {
        type: 'noul',
        noul: 0.982,
        confidence: 0.982
      },
      expense_category: {
        type: 'choice',
        choice: 'cloud_hosting',
        confidence: 0.945,
        probabilities: {
          cloud_hosting: 0.952,
          travel_meals: 0.021,
          office_supplies: 0.018,
          unclear: 0.009
        }
      },
      approval_tier: {
        type: 'score',
        score: 2.0,
        confidence: 0.970,
        probabilities: {
          '0': 0.005,
          '1': 0.025,
          '2': 0.970
        }
      }
    }
  },
  security: {
    id: 'security',
    name: 'Security / Secret',
    description: 'Source code screenshot: detect exposed API keys or tokens and score leak risk severity.',
    draw: drawSecurityPreset,
    spec: {
      title: 'Secret & API Credential Exposure Detection',
      description: 'Detect exposed API tokens, private keys, or passwords in screenshots and code snippets.',
      questions: {
        has_exposed_secret: {
          type: 'noul',
          instructions: 'Does this image contain an exposed plaintext API key, secret token, or credential?',
          criteria: {
            true: 'A live or realistic API key or secret token is visible in plain text',
            false: 'No credentials or private tokens are displayed'
          }
        },
        credential_type: {
          type: 'choice',
          instructions: 'What type of credential or secret is exposed?',
          criteria: {
            api_key: 'Cloud, AI or SaaS API secret token (e.g. sk-proj-...)',
            db_password: 'Database connection string or user password',
            tls_cert: 'Private key certificate or SSH identity',
            none: 'Clean text or code without credentials'
          }
        },
        leak_severity: {
          type: 'score',
          instructions: 'Score the security risk severity of the detected exposure.',
          criteria: [
            'Informational: dummy or sanitized placeholder string',
            'Moderate: internal staging or read-only token',
            'Critical: live production root/admin API key'
          ]
        }
      }
    },
    answers: {
      has_exposed_secret: {
        type: 'noul',
        noul: 0.991,
        confidence: 0.991
      },
      credential_type: {
        type: 'choice',
        choice: 'api_key',
        confidence: 0.932,
        probabilities: {
          api_key: 0.925,
          db_password: 0.055,
          tls_cert: 0.015,
          none: 0.005
        }
      },
      leak_severity: {
        type: 'score',
        score: 1.95,
        confidence: 0.950,
        probabilities: {
          '0': 0.010,
          '1': 0.040,
          '2': 0.950
        }
      }
    }
  },
  navigation: {
    id: 'navigation',
    name: 'Robot Navigation',
    description: 'Indoor robot camera stream: detect forward collision obstacle and choose safe steer maneuver.',
    draw: drawNavigationPreset,
    spec: {
      title: 'Visual Navigation & Robot Obstacle Steering',
      description: 'Analyze robotic camera feed to decide immediate collision avoidance steering maneuvers.',
      questions: {
        collision_risk: {
          type: 'noul',
          instructions: 'Is an obstacle directly obstructing the robot’s immediate forward trajectory?',
          criteria: {
            true: 'Path is obstructed within immediate stopping distance',
            false: 'Forward path is clear of obstacles'
          }
        },
        maneuver: {
          type: 'choice',
          instructions: 'Which navigation maneuver is safe given visual clearance on the left and right?',
          criteria: {
            stop: 'Immediate brake / emergency stop',
            turn_left: 'Steer into clear left corridor',
            turn_right: 'Steer right toward barrier',
            proceed: 'Maintain forward speed'
          }
        },
        obstacle_proximity: {
          type: 'score',
          instructions: 'Estimate obstacle distance to collision horizon.',
          criteria: [
            'Distant (> 5 meters): safe planning horizon',
            'Approaching (2 – 5 meters): initiate diversion',
            'Imminent hazard (< 2 meters): immediate steering or emergency brake required'
          ]
        }
      }
    },
    answers: {
      collision_risk: {
        type: 'noul',
        noul: 0.978,
        confidence: 0.978
      },
      maneuver: {
        type: 'choice',
        choice: 'turn_left',
        confidence: 0.908,
        probabilities: {
          stop: 0.050,
          turn_left: 0.910,
          turn_right: 0.025,
          proceed: 0.015
        }
      },
      obstacle_proximity: {
        type: 'score',
        score: 1.96,
        confidence: 0.960,
        probabilities: {
          '0': 0.010,
          '1': 0.030,
          '2': 0.960
        }
      }
    }
  }
};

/**
 * Validate image decision specification and payload.
 */
export function validateImageSpec(spec, imagePayload) {
  if (!spec || typeof spec !== 'object') {
    throw new Error('Specification must be a JSON object.');
  }
  if (typeof spec.title !== 'string' || !spec.title.trim()) {
    throw new Error('Specification requires a non-empty title string.');
  }
  if (!spec.questions || typeof spec.questions !== 'object' || Object.keys(spec.questions).length === 0) {
    throw new Error('At least one typed question is required.');
  }
  if (!imagePayload || typeof imagePayload !== 'string' || (!imagePayload.startsWith('data:image/') && !imagePayload.startsWith('blob:'))) {
    throw new Error('A valid image (data URL or blob) must be loaded.');
  }
  for (const [id, q] of Object.entries(spec.questions)) {
    if (!/^[a-z][a-z0-9_]*$/.test(id)) {
      throw new Error(`Invalid question ID "${id}": use simple lowercase letters and underscores.`);
    }
    if (!['choice', 'score', 'noul'].includes(q.type)) {
      throw new Error(`Invalid question type for "${id}": must be choice, score, or noul.`);
    }
    if (typeof q.instructions !== 'string' || !q.instructions.trim()) {
      throw new Error(`Question "${id}" requires non-empty instructions.`);
    }
    if (q.type === 'choice') {
      if (!q.criteria || typeof q.criteria !== 'object' || Object.keys(q.criteria).length < 2) {
        throw new Error(`Choice question "${id}" requires at least 2 named criteria options.`);
      }
    }
    if (q.type === 'score') {
      if (!Array.isArray(q.criteria) || q.criteria.length < 2 || q.criteria.length > 10) {
        throw new Error(`Score question "${id}" requires 2 to 10 ordered criteria levels.`);
      }
    }
  }
  return true;
}

/**
 * Execute image decision evaluation across selected engine.
 */
export async function decideImage({ spec, imagePayload, engine = 'client', endpoint = '', key = '', model = 'Imajev-4B' }, signal) {
  validateImageSpec(spec, imagePayload);

  // Global mock interceptor for test suites
  if (globalThis.__MOCK_DECIDE__) {
    return globalThis.__MOCK_DECIDE__(spec, { engine, endpoint, key, model, imagePayload });
  }

  const start = performance.now();

  // Engine 1: Demo Mode / Schema Preview (Simulated Fixture)
  if (engine === 'client') {
    // Check if matching a preset
    let presetKey = null;

    const canonicalize = (obj) => {
      if (Array.isArray(obj)) return obj.map(canonicalize);
      if (obj !== null && typeof obj === 'object') {
        return Object.keys(obj).sort().reduce((acc, k) => {
          acc[k] = canonicalize(obj[k]);
          return acc;
        }, {});
      }
      return obj;
    };

    // The lookup key is canonicalized (order-insensitive keys) and deliberately excludes
    // title and description, so identical questions under different titles share a fixture
    // (intended for an image-agnostic schema preview).
    for (const [k, p] of Object.entries(PRESETS)) {
      if (JSON.stringify(canonicalize(p.spec.questions)) === JSON.stringify(canonicalize(spec.questions))) {
        presetKey = k;
        break;
      }
    }

    let answers;
    if (presetKey && PRESETS[presetKey]) {
      // Use preset simulated fixture outputs
      answers = structuredClone(PRESETS[presetKey].answers);
    } else {
      // Custom image: deterministic illustrative fixture for UI schema validation
      let hash = 0;
      for (let i = 0; i < Math.min(imagePayload.length, 1000); i++) {
        hash = (hash * 31 + imagePayload.charCodeAt(i)) >>> 0;
      }
      answers = {};
      for (const [id, q] of Object.entries(spec.questions)) {
        if (q.type === 'noul') {
          const p = 0.65 + (hash % 300) / 1000;
          answers[id] = { type: 'noul', noul: Math.min(0.99, p), confidence: Math.min(0.99, p) };
        } else if (q.type === 'choice') {
          const keys = Object.keys(q.criteria);
          const pWinner = 0.75 + (hash % 200) / 1000;
          const pOthers = (1 - pWinner) / (keys.length - 1);
          const probabilities = Object.fromEntries(keys.map((k, i) => [k, i === 0 ? pWinner : pOthers]));
          const confidence = (pWinner - 1 / keys.length) / (1 - 1 / keys.length);
          answers[id] = { type: 'choice', choice: keys[0], confidence: Math.max(0, confidence), probabilities };
        } else if (q.type === 'score') {
          const count = q.criteria.length;
          const pLast = 0.80;
          const pRest = (1 - pLast) / (count - 1);
          const probabilities = Object.fromEntries(q.criteria.map((_, i) => [String(i), i === count - 1 ? pLast : pRest]));
          const score = Object.entries(probabilities).reduce((sum, [k, v]) => sum + Number(k) * v, 0);
          const confidence = (pLast - 1 / count) / (1 - 1 / count);
          answers[id] = { type: 'score', score, confidence: Math.max(0, confidence), probabilities };
        }
      }
    }

    // Yield to the browser frame cycle without faking model latency
    await new Promise(r => setTimeout(r, 0));
    const elapsed = performance.now() - start;

    return {
      data: {
        model: 'Demo Mode / Schema Preview (Simulated Fixture)',
        simulated: true,
        simulationNotice: 'Notice: UI and schema validation simulation. Connect a local vision server (e.g. llama.cpp / vLLM / Kev serve) or multimodal API endpoint below to run actual model weights.',
        answers
      },
      elapsed,
      isSimulation: true,
      simulationNotice: 'Notice: UI and schema validation simulation. Connect a local vision server (e.g. llama.cpp / vLLM / Kev serve) or multimodal API endpoint below to run actual model weights.',
      source: 'Demo Mode / Schema Preview (Simulated Fixture)',
      requestedModel: model
    };
  }

  // Engine 2: Local Vision Decision Server (loopback)
  if (engine === 'local') {
    // Same contract as the lab's Kev endpoint: loopback host, http or https, route pinned to
    // /v1/systemone, no credentials, query or fragment. Throws naming the constraint that failed.
    const url = loopbackEndpoint(endpoint);

    const payload = {
      image: imagePayload,
      questions: spec.questions,
      model: model || 'jpt-9b'
    };

    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal
    });
    if (!res.ok) {
      throw new Error(`Local vision server returned HTTP ${res.status}.`);
    }
    // The response is externally derived, so it passes the same validator core.js decide()
    // applies: a wrong type, non-finite probability, bad distribution or out-of-set choice
    // throws naming the question instead of reaching renderImageAnswers.
    // The client engine above is deliberately not gated - it is the disclosed offline fixture.
    const data = validateAnswers(await res.json(), spec.questions);
    return {
      data,
      elapsed: performance.now() - start,
      source: 'Local Vision Decision Server',
      requestedModel: model
    };
  }

  // Engine 3: Multimodal API Adapter (Wity-1 / Gemini / OpenAI)
  if (engine === 'api') {
    if (!key.trim()) {
      throw new Error('Enter an API key for the multimodal adapter.');
    }
    const payload = {
      image: imagePayload,
      questions: spec.questions,
      model: model || 'wity-1'
    };
    const res = await fetch('https://api.typesafe.ai/v1/systemone', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${key.trim()}`
      },
      body: JSON.stringify(payload),
      signal
    });
    if (!res.ok) {
      throw new Error(`Multimodal API returned HTTP ${res.status}. Note: direct browser calls may require CORS configuration or local proxy relay.`);
    }
    // Same boundary as the local engine: the hosted provider's JSON is validated before render.
    const data = validateAnswers(await res.json(), spec.questions);
    return {
      data,
      elapsed: performance.now() - start,
      source: `Multimodal API (${model})`,
      requestedModel: model
    };
  }

  throw new Error(`Unknown engine: ${engine}`);
}

/**
 * Render visual decision answers with decision distributions & meters.
 */
export function renderImageAnswers(target, result, threshold = 0.8) {
  target.replaceChildren();
  if (result.isSimulation || result.data?.simulated) {
    const badge = el('div', undefined, 'notice');
    badge.id = 'simulation-badge';
    badge.style.marginBottom = '1rem';
    badge.append(
      el('strong', 'Notice: UI and schema validation simulation. '),
      'Connect a local vision server (e.g. llama.cpp / vLLM / Kev serve) or multimodal API endpoint below to run actual model weights.'
    );
    target.append(badge);
  }
  for (const [id, a] of Object.entries(result.data.answers)) {
    const box = el('section', undefined, 'answer');
    box.append(
      el('h3', id),
      el('p', a.type === 'choice'
        ? `Selected: ${a.choice}`
        : a.type === 'score'
        ? `Expected score: ${a.score.toFixed(2)}`
        : `p(yes): ${a.noul.toFixed(3)}`)
    );

    const ps = a.type === 'noul' ? { yes: a.noul, no: 1 - a.noul } : a.probabilities;
    for (const [label, value] of Object.entries(ps)) {
      const row = el('div', undefined, 'bar');
      const meter = el('meter');
      meter.min = 0;
      meter.max = 1;
      meter.value = value;
      meter.setAttribute('aria-label', `${label} probability`);
      row.append(el('span', label), meter, el('output', `${(value * 100).toFixed(1)}%`));
      box.append(row);
    }

    if (a.confidence !== undefined) {
      box.append(el('p', `Confidence score: ${Number(a.confidence).toFixed(3)}`));
    }
    box.append(el('p', `Gate verdict: ${gate(a, threshold)}`, 'gate'));
    target.append(box);
  }
}

/**
 * Render JevImageBench v0.1.5 Explorer Table with capability bars & pills.
 */
export function renderBenchTable(container, data) {
  container.replaceChildren();
  if (!data || data.length === 0) {
    container.append(el('p', 'No models match the selected filters.', 'small'));
    return;
  }

  const table = el('table');
  const thead = el('thead');
  const trHead = el('tr');
  ['Rank', 'Model & Base', 'Capability', 'Intelligence', 'Calibration', 'Speed', 'Cost / 1k', 'Source'].forEach(h => {
    trHead.append(el('th', h));
  });
  thead.append(trHead);
  table.append(thead);

  const tbody = el('tbody');
  data.forEach(item => {
    const tr = el('tr');
    if (item.highlight) tr.classList.add('highlight-row');

    // Rank
    tr.append(el('td', `#${item.rank}`));

    // Model & Base + badges
    const tdModel = el('td');
    const strong = el('strong', item.name);
    tdModel.append(strong);
    const sub = el('p', item.baseModel, 'small');
    sub.style.margin = '0.2rem 0';
    tdModel.append(sub);

    const badgeWrap = el('div');
    badgeWrap.style.display = 'flex';
    badgeWrap.style.gap = '0.4rem';
    badgeWrap.style.flexWrap = 'wrap';

    const accessBadge = el('span', item.access, item.access === 'Open Weights' ? 'badge badge-open' : 'badge badge-closed');
    badgeWrap.append(accessBadge);
    if (item.highlight) {
      const topBadge = el('span', 'Top Jev-Class', 'badge badge-top');
      badgeWrap.append(topBadge);
    }
    tdModel.append(badgeWrap);
    tr.append(tdModel);

    // Capability Score with visual bar
    const tdCap = el('td');
    const capVal = el('strong', item.capabilityScore.toFixed(1));
    const capMeter = el('meter');
    capMeter.min = 0;
    capMeter.max = 100;
    capMeter.value = item.capabilityScore;
    capMeter.setAttribute('aria-label', `${item.name} capability score`);
    tdCap.append(capVal, capMeter);
    tr.append(tdCap);

    // Intelligence
    tr.append(el('td', item.intelligence.toFixed(1)));

    // Calibration
    tr.append(el('td', `${item.calibration.toFixed(1)}%`));

    // Speed
    const tdSpeed = el('td');
    const speedPill = el('span', `${item.speed.toFixed(2)}s`, 'pill pill-speed');
    tdSpeed.append(speedPill);
    tr.append(tdSpeed);

    // Cost
    const tdCost = el('td');
    const costText = item.cost < 0.01 ? `$${item.cost.toFixed(4)}` : `$${item.cost.toFixed(3)}`;
    const costPill = el('span', costText, 'pill pill-cost');
    tdCost.append(costPill);
    tr.append(tdCost);

    // Link
    const tdLink = el('td');
    const a = el('a', 'Model card');
    a.href = item.link;
    a.target = '_blank';
    a.rel = 'noreferrer noopener';
    tdLink.append(a);
    tr.append(tdLink);

    tbody.append(tr);
  });
  table.append(tbody);
  container.append(table);
}

/**
 * Initialize Image Lab Interactive Controller
 */
export function setupImageLab() {
  const previewImg = $('preview-image');
  const imageMeta = $('image-meta');
  const questionsArea = $('questions');
  const dropZone = $('drop-zone');
  const fileInput = $('image-file');
  const statusEl = $('status');
  const answersEl = $('answers');
  const thresholdEl = $('threshold');
  const thresholdVal = $('threshold-value');
  const engineSelect = $('engine');
  const localField = $('local-field');
  const apiField = $('api-field');
  const rawDetails = $('raw-details');
  const rawJson = $('raw-json');
  const runBtn = $('run-decision');
  const clearBtn = $('clear-image');

  const benchContainer = $('bench-table-container');
  const benchSearch = $('bench-search');
  const benchAccess = $('bench-access');
  const benchCategory = $('bench-category');
  const benchSort = $('bench-sort');
  const benchCount = $('bench-count');

  let currentImagePayload = '';
  let lastResult = null;
  const offscreenCanvas = document.createElement('canvas');

  function loadImagePayload(dataUrl, infoText = 'Image loaded') {
    currentImagePayload = dataUrl;
    if (previewImg) previewImg.src = dataUrl;
    if (imageMeta) {
      const kb = Math.round(dataUrl.length * 0.75 / 1024);
      imageMeta.textContent = `${infoText} (${kb} KB payload)`;
    }
  }

  function loadPreset(presetId) {
    const p = PRESETS[presetId];
    if (!p) return;
    const dataUrl = p.draw(offscreenCanvas);
    loadImagePayload(dataUrl, `Preset: ${p.name}`);
    if (questionsArea) {
      questionsArea.value = JSON.stringify(p.spec, null, 2);
    }
    if (statusEl) {
      statusEl.textContent = `Preset loaded: ${p.name}. Ready to evaluate.`;
    }
    if (answersEl) answersEl.replaceChildren();
    if (rawDetails) rawDetails.hidden = true;
    lastResult = null;
  }

  // Preset button listeners
  ['ui', 'invoice', 'security', 'navigation'].forEach(id => {
    const btn = $(`preset-${id}`);
    if (btn) {
      btn.addEventListener('click', () => loadPreset(id));
    }
  });

  // File drop zone handling
  if (dropZone) {
    ['dragenter', 'dragover'].forEach(name => {
      dropZone.addEventListener(name, e => {
        e.preventDefault();
        dropZone.classList.add('dragover');
      });
    });
    ['dragleave', 'drop'].forEach(name => {
      dropZone.addEventListener(name, e => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
      });
    });
    dropZone.addEventListener('drop', e => {
      const file = e.dataTransfer?.files?.[0];
      if (file && file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = ev => {
          loadImagePayload(ev.target.result, `Dropped: ${file.name}`);
          if (statusEl) statusEl.textContent = `Custom image dropped: ${file.name}.`;
        };
        reader.readAsDataURL(file);
      }
    });
    dropZone.addEventListener('click', () => {
      fileInput?.click();
    });
  }

  // File input change
  if (fileInput) {
    fileInput.addEventListener('change', () => {
      const file = fileInput.files?.[0];
      if (file && file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = ev => {
          loadImagePayload(ev.target.result, `Selected: ${file.name}`);
          if (statusEl) statusEl.textContent = `Image loaded from file: ${file.name}.`;
        };
        reader.readAsDataURL(file);
      }
    });
  }

  // Paste listener (Ctrl+V / Cmd+V)
  window.addEventListener('paste', e => {
    const items = e.clipboardData?.items;
    if (!items) return;
    for (const item of items) {
      if (item.type.startsWith('image/')) {
        const file = item.getAsFile();
        if (file) {
          const reader = new FileReader();
          reader.onload = ev => {
            loadImagePayload(ev.target.result, 'Pasted from clipboard');
            if (statusEl) statusEl.textContent = 'Pasted image from clipboard.';
          };
          reader.readAsDataURL(file);
          break;
        }
      }
    }
  });

  // Engine change
  if (engineSelect) {
    engineSelect.addEventListener('change', () => {
      const val = engineSelect.value;
      if (localField) localField.hidden = val !== 'local';
      if (apiField) apiField.hidden = val !== 'api';
      const notice = $('engine-notice');
      if (notice) {
        if (val === 'client') {
          notice.hidden = false;
          notice.innerHTML = '<strong>Notice: UI and schema validation simulation.</strong> Connect a local vision server (e.g. llama.cpp / vLLM / Kev serve) or multimodal API endpoint below to run actual model weights.';
        } else if (val === 'local') {
          notice.hidden = false;
          notice.innerHTML = '<strong>Local Vision Server mode:</strong> Connects to your local inference server (e.g. llama.cpp, vLLM, or Kev serve) on loopback for actual model execution.';
        } else if (val === 'api') {
          notice.hidden = false;
          notice.innerHTML = '<strong>Multimodal API mode:</strong> Connects to external vision provider API. Direct browser requests may require CORS headers or local relay.';
        }
      }
    });
  }

  // Threshold slider
  if (thresholdEl) {
    thresholdEl.addEventListener('input', () => {
      const th = Number(thresholdEl.value);
      if (thresholdVal) thresholdVal.textContent = th.toFixed(2);
      if (lastResult && answersEl) {
        renderImageAnswers(answersEl, lastResult, th);
      }
    });
  }

  // Clear button
  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      loadPreset('ui');
    });
  }

  // Run decision button
  if (runBtn) {
    runBtn.addEventListener('click', async () => {
      if (answersEl) answersEl.replaceChildren();
      if (rawDetails) rawDetails.hidden = true;
      lastResult = null;
      let spec;
      try {
        spec = JSON.parse(questionsArea?.value || '{}');
      } catch (err) {
        if (statusEl) statusEl.textContent = `JSON Error: ${err.message}`;
        return;
      }

      const engine = engineSelect ? engineSelect.value : 'client';
      const endpoint = $('endpoint')?.value || 'http://127.0.0.1:8009/v1/systemone';
      const key = $('api-key')?.value || '';
      const model = $('model-name')?.value || 'Imajev-4B';
      const threshold = Number(thresholdEl?.value || 0.8);

      if (statusEl) statusEl.textContent = 'Evaluating visual decision...';
      runBtn.disabled = true;

      try {
        const result = await decideImage({
          spec,
          imagePayload: currentImagePayload,
          engine,
          endpoint,
          key,
          model
        });

        lastResult = result;
        if (statusEl) {
          const simText = result.isSimulation ? ' (Simulated demo — no model runs)' : '';
          statusEl.textContent = `Decision complete${simText}: ${Math.round(result.elapsed)}ms via ${result.source}`;
        }
        if (answersEl) {
          renderImageAnswers(answersEl, result, threshold);
        }
        if (rawDetails && rawJson) {
          rawDetails.hidden = false;
          rawJson.textContent = JSON.stringify({
            request: {
              model: result.requestedModel,
              engine,
              questions: spec.questions,
              imagePayloadBytes: currentImagePayload.length
            },
            response: result.data
          }, null, 2);
        }
      } catch (err) {
        if (statusEl) statusEl.textContent = `Error: ${err.message}`;
      } finally {
        runBtn.disabled = false;
      }
    });
  }

  // Benchmark table filtering and sorting
  function updateBench() {
    if (!benchContainer) return;
    const query = benchSearch?.value || '';
    const access = benchAccess?.value || 'all';
    const category = benchCategory?.value || 'all';
    const sortBy = benchSort?.value || 'capabilityScore';
    const ascending = sortBy === 'speed' || sortBy === 'cost';

    const filtered = filterImageBench(JEV_IMAGE_BENCH_DATA, { query, access, category });
    const sorted = sortImageBench(filtered, sortBy, ascending);

    if (benchCount) {
      benchCount.textContent = `Showing ${sorted.length} of ${JEV_IMAGE_BENCH_DATA.length} systems (v${BENCHMARK_META.version})`;
    }
    renderBenchTable(benchContainer, sorted);
  }

  [benchSearch, benchAccess, benchCategory, benchSort].forEach(el => {
    if (el) {
      el.addEventListener('input', updateBench);
      el.addEventListener('change', updateBench);
    }
  });

  // Initial load
  loadPreset('ui');
  updateBench();
}

// Auto-run if executed in browser context
if (typeof document !== 'undefined') {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setupImageLab);
  } else {
    setupImageLab();
  }
}

