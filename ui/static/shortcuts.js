/**
 * shortcuts.js — keyboard navigation for biss-brain inspector (B1, v0.6)
 *
 * Bindings (only when no input/textarea is focused):
 *   /       focus the search input
 *   j / k   next / prev card (highlights with .is-current)
 *   p       toggle pin on current card
 *   f       forget current card (if forget button present)
 *   o       toggle .expanded on current card
 *   c       toggle compose checkbox on current card
 *   g s     navigate to /sessions
 *   g t     navigate to /timeline
 *   g x     navigate to /scatter
 *   g r     navigate to /graph
 *   g h     navigate to / (home)
 *   Escape  clear focus / close compose drawer
 *
 * Compose bar: appears when any .compose-pick checkbox is checked.
 */
(function () {
  "use strict";

  // ── Helpers ────────────────────────────────────────────────────────────────

  function _isTyping(e) {
    var tag = (e.target.tagName || "").toUpperCase();
    return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
  }

  function _projectFromUrl() {
    var params = new URLSearchParams(window.location.search);
    return params.get("project") || "";
  }

  function _cards() {
    return Array.from(document.querySelectorAll(".card"));
  }

  function _currentIndex() {
    var cards = _cards();
    for (var i = 0; i < cards.length; i++) {
      if (cards[i].classList.contains("is-current")) return i;
    }
    return -1;
  }

  function _setCurrentIndex(idx) {
    var cards = _cards();
    cards.forEach(function (c) { c.classList.remove("is-current"); });
    if (idx < 0 || idx >= cards.length) return null;
    var card = cards[idx];
    card.classList.add("is-current");
    card.scrollIntoView({ block: "nearest" });
    return card;
  }

  function _navigate(delta) {
    var cards = _cards();
    if (!cards.length) return;
    var idx = _currentIndex();
    var next;
    if (idx < 0) {
      next = delta > 0 ? 0 : cards.length - 1;
    } else {
      next = idx + delta;
      if (next < 0) next = 0;
      if (next >= cards.length) next = cards.length - 1;
    }
    _setCurrentIndex(next);
  }

  function _currentCard() {
    var idx = _currentIndex();
    if (idx < 0) return null;
    return _cards()[idx];
  }

  function _goTo(path) {
    var project = _projectFromUrl();
    window.location.href = project
      ? path + "?project=" + encodeURIComponent(project)
      : path;
  }

  // ── Compose bar ────────────────────────────────────────────────────────────

  function _updateComposeBar() {
    var bar = document.getElementById("compose-bar");
    if (!bar) return;
    var checked = document.querySelectorAll(".compose-pick:checked");
    var count = checked.length;
    if (count > 0) {
      bar.classList.add("visible");
      var countEl = bar.querySelector(".count");
      if (countEl) countEl.textContent = count + " selected";
    } else {
      bar.classList.remove("visible");
    }
  }

  function _composeSelected() {
    var checked = document.querySelectorAll(".compose-pick:checked");
    var ids = Array.from(checked).map(function (el) {
      return el.getAttribute("data-id");
    }).filter(Boolean);
    if (!ids.length) return;

    var project = _projectFromUrl();
    var url = "/api/compose?project=" + encodeURIComponent(project);

    // Remove existing drawer if present
    var existing = document.getElementById("compose-drawer");
    if (existing) existing.remove();

    var payload = JSON.stringify({ ids: ids });

    // Prefer htmx.ajax if available
    if (window.htmx && typeof window.htmx.ajax === "function") {
      window.htmx.ajax("POST", url, {
        values: payload,
        headers: { "Content-Type": "application/json" },
        target: "body",
        swap: "beforeend",
      });
    } else {
      fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: payload,
      })
        .then(function (r) { return r.text(); })
        .then(function (html) {
          var tmp = document.createElement("div");
          tmp.innerHTML = html;
          while (tmp.firstChild) {
            document.body.appendChild(tmp.firstChild);
          }
        })
        .catch(function (err) {
          console.error("compose fetch failed:", err);
        });
    }
  }

  // ── g-prefix sequence state ────────────────────────────────────────────────

  var _pendingG = false;
  var _gTimeout = null;

  function _clearG() {
    _pendingG = false;
    if (_gTimeout) { clearTimeout(_gTimeout); _gTimeout = null; }
  }

  // ── Keyboard handler ───────────────────────────────────────────────────────

  document.addEventListener("keydown", function (e) {
    // Always handle Escape regardless of focus
    if (e.key === "Escape") {
      var drawer = document.getElementById("compose-drawer");
      if (drawer) { drawer.remove(); return; }
      if (document.activeElement) document.activeElement.blur();
      _clearG();
      return;
    }

    // Don't intercept when typing in inputs
    if (_isTyping(e)) { _clearG(); return; }

    // ── g-sequence ──────────────────────────────────────────────────────────
    if (_pendingG) {
      _clearG();
      switch (e.key) {
        case "s": e.preventDefault(); _goTo("/sessions"); return;
        case "t": e.preventDefault(); _goTo("/timeline"); return;
        case "x": e.preventDefault(); _goTo("/scatter"); return;
        case "r": e.preventDefault(); _goTo("/graph"); return;
        case "h": e.preventDefault(); _goTo("/"); return;
      }
      return;
    }

    if (e.key === "g" && !e.ctrlKey && !e.metaKey && !e.altKey) {
      e.preventDefault();
      _pendingG = true;
      _gTimeout = setTimeout(_clearG, 1500);
      return;
    }

    // ── Single-key bindings ─────────────────────────────────────────────────
    switch (e.key) {
      case "/":
        e.preventDefault();
        var searchInput = document.querySelector('input[name="q"]');
        if (searchInput) searchInput.focus();
        return;

      case "j":
        e.preventDefault();
        _navigate(1);
        return;

      case "k":
        e.preventDefault();
        _navigate(-1);
        return;

      case "p": {
        e.preventDefault();
        var card = _currentCard();
        if (!card) return;
        // Find pin/unpin button (contains 'pin' text)
        var btns = Array.from(card.querySelectorAll("button"));
        var pinBtn = btns.find(function (b) {
          var t = (b.textContent || "").trim().toLowerCase();
          return t === "pin" || t === "unpin";
        });
        if (pinBtn) pinBtn.click();
        return;
      }

      case "f": {
        e.preventDefault();
        var card = _currentCard();
        if (!card) return;
        var btns = Array.from(card.querySelectorAll("button"));
        var forgetBtn = btns.find(function (b) {
          return (b.textContent || "").trim().toLowerCase() === "forget";
        });
        if (forgetBtn) forgetBtn.click();
        return;
      }

      case "o": {
        e.preventDefault();
        var card = _currentCard();
        if (card) card.classList.toggle("expanded");
        return;
      }

      case "c": {
        e.preventDefault();
        var card = _currentCard();
        if (!card) return;
        var cb = card.querySelector("input.compose-pick");
        if (cb) {
          cb.checked = !cb.checked;
          cb.dispatchEvent(new Event("change", { bubbles: true }));
        }
        return;
      }
    }
  });

  // ── Compose bar event delegation ───────────────────────────────────────────

  document.addEventListener("change", function (e) {
    if (e.target && e.target.classList.contains("compose-pick")) {
      _updateComposeBar();
    }
  });

  document.addEventListener("click", function (e) {
    var btn = e.target && e.target.closest("#compose-go");
    if (btn) {
      e.preventDefault();
      _composeSelected();
    }
  });

  // Initial bar state in case of page reload with checkboxes pre-checked
  document.addEventListener("DOMContentLoaded", _updateComposeBar);

})();
