/* UltraTex project page — minimal interactions */

// Copy BibTeX to clipboard
(function () {
  var btn = document.getElementById('copyBib');
  var pre = document.getElementById('bibtex');
  if (!btn || !pre) return;

  btn.addEventListener('click', function () {
    var text = pre.textContent;

    function flash(label) {
      var orig = 'Copy';
      btn.textContent = label;
      btn.classList.add('done');
      setTimeout(function () {
        btn.textContent = orig;
        btn.classList.remove('done');
      }, 1600);
    }

    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(
        function () { flash('Copied'); },
        function () { fallback(); }
      );
    } else {
      fallback();
    }

    function fallback() {
      // Older browsers / non-secure origins: select the text so the user can copy it.
      var range = document.createRange();
      range.selectNodeContents(pre);
      var sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
      flash('Selected');
    }
  });
})();

// Highlight the nav link for the section currently in view
(function () {
  var links = Array.prototype.slice.call(
    document.querySelectorAll('.nav-links a[href^="#"]')
  );
  if (!links.length || !('IntersectionObserver' in window)) return;

  var byId = {};
  links.forEach(function (a) {
    var id = a.getAttribute('href').slice(1);
    var el = document.getElementById(id);
    if (el) byId[id] = a;
  });

  var current = null;
  var observer = new IntersectionObserver(
    function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        var a = byId[entry.target.id];
        if (!a || a === current) return;
        if (current) current.style.color = '';
        a.style.color = 'var(--fg)';
        current = a;
      });
    },
    { rootMargin: '-56px 0px -70% 0px', threshold: 0 }
  );

  Object.keys(byId).forEach(function (id) {
    observer.observe(document.getElementById(id));
  });
})();

