document.addEventListener('DOMContentLoaded', () => {
  const CONTACT_EMAIL = 'goleaaryan7@gmail.com';
  const CONTACT_TIMEOUT_MS = 15000;
  const CHAT_TIMEOUT_MS = 12000;
  const STACKED_LAYOUT = window.matchMedia('(min-width: 1025px), (min-width: 821px) and (min-height: 501px)');

  const nav = document.querySelector('.global-nav');
  const isStackedLayout = () => STACKED_LAYOUT.matches;

  // Only reachable during local testing: the deployed site is never opened as a file.
  const OPENED_AS_FILE = window.location.protocol === 'file:';
  const OPENED_AS_FILE_MESSAGE = 'This page was opened as a file, so it can\'t reach the server. Run "npm run dev" and open http://localhost:8888 instead.';

  initFadeIn();
  initSmoothAnchors();
  initTheme();
  initPreloader();
  initToastAndCopy();
  initNavbar();
  initScrollSpy();
  initScrollProgress();
  initPointerEffects();
  const videos = initVideos();
  initProjects(videos);
  initContactForm();
  initChatbot();
  initCursor();
  initClock();

  function initFadeIn() {
    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduceMotion) return;

    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.style.opacity = '1';
        entry.target.style.transform = 'translateY(0)';
        observer.unobserve(entry.target);
      });
    }, { rootMargin: '0px 0px -60px 0px', threshold: 0.05 });

    document.querySelectorAll('.section-content, .timeline-item, .certifications').forEach((element) => {
      element.style.opacity = '0';
      element.style.transform = 'translateY(30px)';
      element.style.transition = 'opacity 0.6s ease-out, transform 0.6s ease-out';
      observer.observe(element);
    });
  }

  function initSmoothAnchors() {
    document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
      anchor.addEventListener('click', (event) => {
        const target = document.querySelector(anchor.getAttribute('href'));
        if (!target) return;
        event.preventDefault();
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      });
    });
  }

  function initTheme() {
    const toggle = document.getElementById('themeToggle');

    // The switch starts in the saved theme without animating into it.
    document.body.classList.add('theme-init');
    toggle.checked = !document.body.classList.contains('light-mode');
    requestAnimationFrame(() => requestAnimationFrame(() => document.body.classList.remove('theme-init')));

    // The browser bar on iPhone and Android takes the page colour.
    const themeColor = document.getElementById('themeColor');
    const syncThemeColor = () => {
      themeColor.content = document.body.classList.contains('light-mode') ? '#f8f9fa' : '#080b16';
    };
    syncThemeColor();

    toggle.addEventListener('change', () => {
      const isLight = !toggle.checked;
      document.body.classList.toggle('light-mode', isLight);
      syncThemeColor();
      try {
        localStorage.setItem('theme', isLight ? 'light' : 'dark');
      } catch {
        // The choice still applies to this visit when storage is blocked.
      }
    });
  }

  function initPreloader() {
    const preloader = document.querySelector('.preloader');
    const hide = () => preloader.classList.add('done');
    window.addEventListener('load', hide, { once: true });
    // Slow fonts or icons from a CDN should never hold the page back.
    setTimeout(hide, 1500);
  }

  function initToastAndCopy() {
    const toast = document.getElementById('toast');
    let timer = null;
    const show = (text) => {
      toast.textContent = text;
      toast.hidden = false;
      requestAnimationFrame(() => toast.classList.add('visible'));
      clearTimeout(timer);
      timer = setTimeout(() => toast.classList.remove('visible'), 2600);
    };

    document.querySelectorAll('[data-copy]').forEach((button) => {
      button.addEventListener('click', async () => {
        const value = button.dataset.copy;
        try {
          await navigator.clipboard.writeText(value);
          show(`Discord username copied: ${value}`);
        } catch {
          show(`Discord: ${value}`);
        }
      });
    });
  }

  function initNavbar() {
    const logo = document.querySelector('.logo');
    const projectsSection = document.getElementById('projects');
    let projectsInView = false;

    const syncScrollState = () => {
      logo.classList.toggle('compact', window.scrollY > 50);
      nav.classList.toggle('scrolled', window.scrollY > 20);
    };

    // The stacked project cards fill the screen, so the navbar steps aside for them.
    const syncVisibility = () => {
      const hidden = projectsInView && isStackedLayout();
      nav.style.transform = hidden ? 'translateY(calc(-100% - 24px))' : 'translateY(0)';
      nav.style.opacity = hidden ? '0' : '1';
      nav.style.pointerEvents = hidden ? 'none' : 'all';
    };

    new IntersectionObserver(([entry]) => {
      projectsInView = entry.isIntersecting;
      syncVisibility();
    }, { rootMargin: '-1px 0px -99% 0px' }).observe(projectsSection);

    window.addEventListener('scroll', syncScrollState, { passive: true });
    window.addEventListener('resize', syncVisibility);
    syncScrollState();
  }

  function initScrollSpy() {
    const container = document.querySelector('.nav-links');
    const links = [...document.querySelectorAll('.nav-item')];
    const indicator = document.createElement('span');
    indicator.className = 'nav-indicator';
    indicator.setAttribute('aria-hidden', 'true');
    container.prepend(indicator);

    // The indicator slides to a hovered link, then back to the current section.
    const moveIndicator = (link) => {
      if (!link) {
        indicator.style.opacity = '0';
        return;
      }
      indicator.style.width = `${link.offsetWidth}px`;
      indicator.style.transform = `translate(${link.offsetLeft}px, ${link.offsetTop + link.offsetHeight / 2}px) translateY(-50%)`;
      indicator.style.opacity = '1';
    };
    const activeLink = () => links.find((link) => link.classList.contains('nav-active'));

    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        links.forEach((link) => {
          link.classList.toggle('nav-active', link.getAttribute('href') === `#${entry.target.id}`);
        });
        moveIndicator(activeLink());
      });
    }, { rootMargin: '-20% 0px -60% 0px', threshold: 0 });

    document.querySelectorAll('section[id]').forEach((section) => observer.observe(section));
    links.forEach((link) => link.addEventListener('mouseenter', () => moveIndicator(link)));
    container.addEventListener('mouseleave', () => moveIndicator(activeLink()));
    window.addEventListener('resize', () => moveIndicator(activeLink()));
    document.fonts?.ready.then(() => moveIndicator(activeLink()));
  }

  function initScrollProgress() {
    const bar = document.querySelector('.scroll-progress');
    let frame = null;
    const update = () => {
      frame = null;
      const max = document.documentElement.scrollHeight - window.innerHeight;
      bar.style.transform = `scaleX(${max > 0 ? Math.min(1, window.scrollY / max) : 0})`;
    };
    window.addEventListener('scroll', () => {
      if (!frame) frame = requestAnimationFrame(update);
    }, { passive: true });
    update();
  }

  // Pointer effects: card spotlights, the hero grid glow, and the photo tilt.
  function initPointerEffects() {
    const finePointer = window.matchMedia('(hover: hover) and (pointer: fine)').matches;
    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (!finePointer) return;

    const SPOTLIGHT = '.system-card, .skill-bento-card, .timeline-content, .certifications, .contact-form';
    document.addEventListener('pointermove', (event) => {
      const card = event.target.closest?.(SPOTLIGHT);
      if (!card) return;
      const rect = card.getBoundingClientRect();
      card.style.setProperty('--mx', `${event.clientX - rect.left}px`);
      card.style.setProperty('--my', `${event.clientY - rect.top}px`);
    }, { passive: true });

    const hero = document.querySelector('.hero');
    const photo = document.querySelector('.hero-pic .image-clip');
    hero.addEventListener('pointermove', (event) => {
      const rect = hero.getBoundingClientRect();
      hero.style.setProperty('--hx', `${event.clientX - rect.left}px`);
      hero.style.setProperty('--hy', `${event.clientY - rect.top}px`);
      hero.classList.add('pointer-active');
      if (reduceMotion || !photo) return;
      const box = photo.getBoundingClientRect();
      const x = (event.clientX - (box.left + box.width / 2)) / rect.width;
      const y = (event.clientY - (box.top + box.height / 2)) / rect.height;
      photo.style.setProperty('--tilt-y', `${(x * 10).toFixed(2)}deg`);
      photo.style.setProperty('--tilt-x', `${(-y * 10).toFixed(2)}deg`);
    }, { passive: true });
    hero.addEventListener('pointerleave', () => {
      hero.classList.remove('pointer-active');
      photo?.style.setProperty('--tilt-x', '0deg');
      photo?.style.setProperty('--tilt-y', '0deg');
    });
  }

  function initVideos() {
    const containers = [...document.querySelectorAll('.video-container[data-video]')];
    const IDLE_MS = 2500;
    const SEEK_STEP = 5;

    const formatTime = (seconds) => {
      if (!Number.isFinite(seconds)) return '0:00';
      const minutes = Math.floor(seconds / 60);
      return `${minutes}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`;
    };

    const pauseAll = (except) => {
      containers.forEach((container) => {
        if (container === except) return;
        const video = container.querySelector('video');
        if (!video.paused) video.pause();
      });
    };

    const setIcon = (button, icon, label) => {
      button.querySelector('i').className = `bx ${icon}`;
      button.setAttribute('aria-label', label);
    };

    function buildControls(container, video) {
      const bar = document.createElement('div');
      bar.className = 'vp-controls';
      bar.innerHTML = `
        <button type="button" class="vp-btn vp-play" aria-label="Pause"><i class="bx bx-pause"></i></button>
        <div class="vp-progress" role="slider" tabindex="0" aria-label="Seek" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0">
          <div class="vp-track"><div class="vp-buffered"></div><div class="vp-fill"></div></div>
          <div class="vp-thumb"></div>
        </div>
        <span class="vp-time">0:00 / 0:00</span>
        <button type="button" class="vp-btn vp-mute" aria-label="Unmute"><i class="bx bx-volume-mute"></i></button>
        <button type="button" class="vp-btn vp-fs" aria-label="Full screen"><i class="bx bx-fullscreen"></i></button>`;
      container.append(bar);

      const playButton = bar.querySelector('.vp-play');
      const muteButton = bar.querySelector('.vp-mute');
      const fullscreenButton = bar.querySelector('.vp-fs');
      const progress = bar.querySelector('.vp-progress');
      const fill = bar.querySelector('.vp-fill');
      const buffered = bar.querySelector('.vp-buffered');
      const thumb = bar.querySelector('.vp-thumb');
      const time = bar.querySelector('.vp-time');
      let idleTimer = null;

      const togglePlay = () => (video.paused ? video.play().catch(() => {}) : video.pause());

      const wake = () => {
        container.classList.remove('controls-idle');
        clearTimeout(idleTimer);
        idleTimer = setTimeout(() => {
          if (!video.paused && !container.classList.contains('scrubbing')) container.classList.add('controls-idle');
        }, IDLE_MS);
      };

      const renderTime = () => {
        const ratio = video.duration ? video.currentTime / video.duration : 0;
        fill.style.width = `${ratio * 100}%`;
        thumb.style.left = `${ratio * 100}%`;
        time.textContent = `${formatTime(video.currentTime)} / ${formatTime(video.duration)}`;
        progress.setAttribute('aria-valuenow', String(Math.round(ratio * 100)));
        progress.setAttribute('aria-valuetext', time.textContent);
      };

      const renderBuffered = () => {
        if (!video.duration || !video.buffered.length) return;
        buffered.style.width = `${(video.buffered.end(video.buffered.length - 1) / video.duration) * 100}%`;
      };

      const seekTo = (clientX) => {
        if (!video.duration) return;
        const rect = progress.getBoundingClientRect();
        const ratio = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
        video.currentTime = ratio * video.duration;
        renderTime();
      };

      playButton.addEventListener('click', togglePlay);
      muteButton.addEventListener('click', () => {
        video.muted = !video.muted;
        if (!video.muted && video.volume === 0) video.volume = 1;
      });
      fullscreenButton.addEventListener('click', () => {
        if (document.fullscreenElement) {
          document.exitFullscreen();
        } else if (container.requestFullscreen) {
          container.requestFullscreen().catch(() => video.webkitEnterFullscreen?.());
        } else {
          // iPhone Safari can only make the video itself full screen.
          video.webkitEnterFullscreen?.();
        }
      });

      progress.addEventListener('pointerdown', (event) => {
        progress.setPointerCapture(event.pointerId);
        container.classList.add('scrubbing');
        seekTo(event.clientX);
      });
      progress.addEventListener('pointermove', (event) => {
        if (container.classList.contains('scrubbing')) seekTo(event.clientX);
      });
      const endScrub = () => {
        container.classList.remove('scrubbing');
        wake();
      };
      progress.addEventListener('pointerup', endScrub);
      progress.addEventListener('pointercancel', endScrub);
      progress.addEventListener('keydown', (event) => {
        const moves = { ArrowRight: SEEK_STEP, ArrowLeft: -SEEK_STEP };
        if (event.key in moves) video.currentTime = Math.max(0, video.currentTime + moves[event.key]);
        else if (event.key === 'Home') video.currentTime = 0;
        else if (event.key === 'End' && video.duration) video.currentTime = video.duration - 1;
        else if (event.key === ' ' || event.key === 'k') togglePlay();
        else return;
        event.preventDefault();
        wake();
      });

      // Mouse: a click on the video plays or pauses. Touch: a tap shows or hides the controls.
      video.addEventListener('pointerup', (event) => {
        if (event.pointerType === 'mouse') {
          togglePlay();
          wake();
        } else if (container.classList.contains('controls-idle')) {
          wake();
        } else if (!video.paused) {
          container.classList.add('controls-idle');
        }
      });
      container.addEventListener('pointermove', (event) => {
        if (event.pointerType === 'mouse') wake();
      });
      bar.addEventListener('pointerdown', wake);

      video.addEventListener('play', () => {
        setIcon(playButton, 'bx-pause', 'Pause');
        container.classList.remove('is-paused');
        wake();
      });
      video.addEventListener('pause', () => {
        setIcon(playButton, 'bx-play', 'Play');
        container.classList.add('is-paused');
        container.classList.remove('controls-idle');
      });
      video.addEventListener('volumechange', () => {
        const muted = video.muted || video.volume === 0;
        setIcon(muteButton, muted ? 'bx-volume-mute' : 'bx-volume-full', muted ? 'Unmute' : 'Mute');
      });
      video.addEventListener('timeupdate', renderTime);
      video.addEventListener('loadedmetadata', renderTime);
      video.addEventListener('progress', renderBuffered);
      document.addEventListener('fullscreenchange', () => {
        const active = document.fullscreenElement === container;
        setIcon(fullscreenButton, active ? 'bx-exit-fullscreen' : 'bx-fullscreen', active ? 'Exit full screen' : 'Full screen');
      });
      renderTime();
    }

    containers.forEach((container) => {
      const video = container.querySelector('video');
      const overlay = container.querySelector('.play-overlay-v2');
      const label = overlay.querySelector('.play-label');
      const poster = container.querySelector('.showcase-img');
      let defaultLabel = label.textContent;
      let controlsBuilt = false;

      video.controls = false;

      const loadSource = () => {
        if (!video.getAttribute('src') || video.error) video.src = container.dataset.video;
      };

      const showUnavailable = () => {
        container.classList.remove('playing', 'buffering');
        label.textContent = 'Could not load · tap to retry';
      };

      // The label shows the demo length once its details have loaded.
      video.addEventListener('loadedmetadata', () => {
        defaultLabel = `Watch demo · ${formatTime(video.duration)}`;
        if (!container.classList.contains('playing')) label.textContent = defaultLabel;
      });

      overlay.addEventListener('click', () => {
        pauseAll(container);
        label.textContent = defaultLabel;
        if (poster) video.poster = poster.currentSrc || poster.src;
        loadSource();
        if (!controlsBuilt) {
          buildControls(container, video);
          controlsBuilt = true;
        }
        container.classList.add('playing', 'buffering');
        video.play().catch((error) => {
          if (error.name !== 'AbortError') showUnavailable();
        });
      });

      video.addEventListener('playing', () => container.classList.remove('buffering'));
      video.addEventListener('pause', () => container.classList.remove('buffering'));
      video.addEventListener('waiting', () => container.classList.add('buffering'));
      video.addEventListener('error', showUnavailable);
      video.addEventListener('play', () => pauseAll(container));
    });

    // Fetch only each video's header when it nears the screen, to show its length.
    const nearby = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        const video = entry.target.querySelector('video');
        if (!video.getAttribute('src')) {
          video.preload = 'metadata';
          video.src = entry.target.dataset.video;
        }
        nearby.unobserve(entry.target);
      });
    }, { rootMargin: '400px 0px' });

    const visibility = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) return;
        const video = entry.target.querySelector('video');
        if (!video.paused) video.pause();
      });
    }, { threshold: 0.2 });
    containers.forEach((container) => {
      nearby.observe(container);
      visibility.observe(container);
    });

    return { pauseAll };
  }

  function initProjects(videos) {
    const list = document.querySelector('.projects-list');
    const cards = [...document.querySelectorAll('.project-showcase')];
    const sideItems = [...document.querySelectorAll('.nav-item-wrapper')];
    const filterButtons = [...document.querySelectorAll('.filter-btn')];
    let visibleCards = cards;
    let activeId = null;
    let layoutFrame = null;

    const listTop = () => list.getBoundingClientRect().top + window.scrollY;

    // Sticky cards report their stuck position, so document positions come from card heights.
    const cardTops = () => {
      let top = listTop();
      return visibleCards.map((card) => {
        top += parseFloat(getComputedStyle(card).marginTop) || 0;
        const cardTop = top;
        top += card.offsetHeight;
        return cardTop;
      });
    };

    // Where a card sticks: below the navbar on phones, or higher if it is taller than the screen.
    const stickLimit = () => (isStackedLayout() ? 0 : nav.getBoundingClientRect().bottom + 10);

    // A card taller than the screen scrolls to its bottom edge before it sticks.
    const updateStickyOffsets = () => {
      visibleCards.forEach((card) => {
        const offset = Math.min(stickLimit(), window.innerHeight - card.offsetHeight);
        card.style.setProperty('--stick-top', `${offset}px`);
      });
    };

    const syncActiveProject = () => {
      if (!visibleCards.length) return;
      const probe = window.scrollY + window.innerHeight * 0.4;
      let activeIndex = 0;
      cardTops().forEach((top, index) => {
        if (top <= probe) activeIndex = index;
      });

      const activeCard = visibleCards[activeIndex];
      const id = activeCard.dataset.project;
      if (id === activeId) return;

      activeId = id;
      sideItems.forEach((item) => item.classList.toggle('active', item.dataset.project === id));
      // A stuck card stays "in view" beneath the next one, so its video is paused here.
      if (getComputedStyle(activeCard).position === 'sticky') videos.pauseAll(activeCard.querySelector('.video-container'));
    };

    const scheduleLayoutUpdate = () => {
      if (layoutFrame) return;
      layoutFrame = requestAnimationFrame(() => {
        layoutFrame = null;
        updateStickyOffsets();
        syncActiveProject();
      });
    };

    const applyFilter = (filter) => {
      filterButtons.forEach((button) => {
        const isActive = button.dataset.filter === filter;
        button.classList.toggle('active', isActive);
        button.setAttribute('aria-pressed', String(isActive));
      });

      visibleCards = cards.filter((card) => {
        if (filter === 'all') return true;
        return filter === 'top' ? card.dataset.priority === 'top' : card.dataset.category === filter;
      });
      cards.forEach((card) => {
        card.style.display = visibleCards.includes(card) ? '' : 'none';
      });

      let position = 0;
      const visibleIds = new Set(visibleCards.map((card) => card.dataset.project));
      const orderedItems = cards
        .map((card) => sideItems.find((item) => item.dataset.project === card.dataset.project));
      orderedItems.forEach((item) => {
        const isVisible = visibleIds.has(item.dataset.project);
        item.style.display = isVisible ? '' : 'none';
        if (isVisible) {
          position += 1;
          item.querySelector('.nav-no').textContent = String(position).padStart(2, '0');
        }
      });

      activeId = null;
      videos.pauseAll();
      updateStickyOffsets();
    };

    filterButtons.forEach((button) => {
      button.addEventListener('click', () => {
        applyFilter(button.dataset.filter);
        if (window.scrollY > listTop()) {
          window.scrollTo({ top: listTop(), behavior: 'auto' });
        }
        syncActiveProject();
      });
    });

    const scrollToProject = (id) => {
      const index = visibleCards.findIndex((card) => card.dataset.project === id);
      if (index === -1) return;
      const navOffset = stickLimit();
      window.scrollTo({ top: cardTops()[index] - navOffset, behavior: 'smooth' });
    };

    sideItems.forEach((item) => {
      item.addEventListener('click', () => scrollToProject(item.dataset.project));
      item.addEventListener('keydown', (event) => {
        if (event.key !== 'Enter' && event.key !== ' ') return;
        event.preventDefault();
        scrollToProject(item.dataset.project);
      });
    });

    const initialFilter = document.querySelector('.filter-btn.active')?.dataset.filter || 'all';
    applyFilter(initialFilter);
    syncActiveProject();

    const resizeObserver = new ResizeObserver(scheduleLayoutUpdate);
    cards.forEach((card) => resizeObserver.observe(card));
    window.addEventListener('resize', scheduleLayoutUpdate);
    window.addEventListener('scroll', scheduleLayoutUpdate, { passive: true });
  }

  function initContactForm() {
    const form = document.getElementById('contactForm');
    const submitButton = document.getElementById('submitBtn');
    const status = document.getElementById('formMessage');
    let hideTimer = null;

    const showStatus = (type, text) => {
      clearTimeout(hideTimer);
      status.className = `form-message ${type}`;
      status.textContent = text;
      status.hidden = false;
    };

    const showFailure = (name, message) => {
      showStatus('error', "Your message couldn't be sent. Your text is still in the form, or you can ");
      const link = document.createElement('a');
      const subject = encodeURIComponent(`Portfolio enquiry from ${name}`);
      link.href = `mailto:${CONTACT_EMAIL}?subject=${subject}&body=${encodeURIComponent(message)}`;
      link.textContent = 'send it by email instead';
      status.append(link, '.');
    };

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const formData = new FormData(form);

      submitButton.disabled = true;
      submitButton.textContent = 'Sending...';
      showStatus('info', 'Sending your message...');

      try {
        const response = await fetch('/', {
          method: 'POST',
          headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
          body: new URLSearchParams(formData).toString(),
          signal: AbortSignal.timeout(CONTACT_TIMEOUT_MS)
        });
        if (!response.ok) throw new Error(`Form endpoint returned ${response.status}`);

        form.reset();
        showStatus('success', "Message sent! I'll get back to you soon.");
        hideTimer = setTimeout(() => {
          status.hidden = true;
        }, 6000);
      } catch (error) {
        console.error('Contact form failed:', error);
        if (OPENED_AS_FILE) {
          showStatus('error', OPENED_AS_FILE_MESSAGE);
        } else {
          showFailure(formData.get('name'), formData.get('message'));
        }
      } finally {
        submitButton.disabled = false;
        submitButton.textContent = 'Send Message';
      }
    });
  }

  function initChatbot() {
    const button = document.getElementById('chatbotButton');
    const panel = document.getElementById('chatbotWindow');
    const closeButton = document.getElementById('chatbotClose');
    const messages = document.getElementById('chatbotMessages');
    const input = document.getElementById('chatbotInput');
    const sendButton = document.getElementById('chatbotSend');

    const SECTIONS = {
      home: 'home',
      about: 'about',
      focus: 'systems',
      systems: 'systems',
      skills: 'skills',
      experience: 'experience',
      education: 'experience',
      projects: 'projects',
      contact: 'contact'
    };
    const NAVIGATION_PATTERN = /^(?:(?:please\s+)?(?:go to|take me to|navigate to|jump to|scroll to|open|show(?: me)?)\s+)?(?:the\s+|your\s+|his\s+)?(home|about|focus|systems|skills|experience|education|projects|contact)(?:\s+(?:section|page))?[\s.!]*$/i;
    const GREETING_PATTERN = /^(?:hi|hello|hey|hii+|greetings|namaste)(?:\s+there)?[\s.!]*$/i;

    const openPanel = () => {
      panel.classList.add('active');
      document.body.classList.add('chat-open');
      button.style.display = 'none';
      // On phones the keyboard would cover the sheet as it opens, so focus waits for a tap.
      if (window.matchMedia('(min-width: 601px)').matches) input.focus();
    };

    const closePanel = () => {
      panel.classList.remove('active');
      document.body.classList.remove('chat-open');
      button.style.display = 'flex';
    };

    const addMessage = (text, sender) => {
      const wrapper = document.createElement('div');
      wrapper.className = `message ${sender}-message`;
      const content = document.createElement('div');
      content.className = 'message-content';
      const paragraph = document.createElement('p');
      paragraph.textContent = text;
      content.append(paragraph);
      wrapper.append(content);
      messages.append(wrapper);
      messages.scrollTop = messages.scrollHeight;
      return { wrapper, paragraph };
    };

    const navigateTo = (sectionId) => {
      const target = document.getElementById(sectionId);
      if (!target) return;
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      setTimeout(closePanel, 800);
    };

    const askAssistant = async (question) => {
      const pending = addMessage('', 'bot');
      pending.wrapper.classList.add('pending');
      pending.paragraph.innerHTML = '<span class="loader4 loader-sm" role="img" aria-label="Thinking"><span></span><span></span><span></span><span></span></span>';
      sendButton.disabled = true;

      try {
        const response = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: question }),
          signal: AbortSignal.timeout(CHAT_TIMEOUT_MS)
        });
        const data = await response.json();
        pending.paragraph.textContent = data.message || data.error || 'Could you rephrase that?';
      } catch (error) {
        console.error('Chat failed:', error);
        pending.paragraph.textContent = OPENED_AS_FILE
          ? OPENED_AS_FILE_MESSAGE
          : `I'm having trouble connecting right now. You can reach Aaryan at ${CONTACT_EMAIL}.`;
      } finally {
        pending.wrapper.classList.remove('pending');
        sendButton.disabled = false;
        messages.scrollTop = messages.scrollHeight;
      }
    };

    const handleMessage = () => {
      const question = input.value.trim();
      if (!question || sendButton.disabled) return;
      addMessage(question, 'user');
      input.value = '';

      if (GREETING_PATTERN.test(question)) {
        addMessage("Hi there! I'm Aaryan's portfolio assistant. Ask me about hiring him, freelance web projects, or his skills and experience.", 'bot');
        return;
      }

      const navigation = question.match(NAVIGATION_PATTERN);
      if (navigation) {
        addMessage(`Taking you to ${navigation[1].toLowerCase()}...`, 'bot');
        navigateTo(SECTIONS[navigation[1].toLowerCase()]);
        return;
      }

      askAssistant(question);
    };

    button.addEventListener('click', openPanel);
    closeButton.addEventListener('click', closePanel);
    sendButton.addEventListener('click', handleMessage);
    input.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') handleMessage();
    });

    document.querySelectorAll('.suggest-chip').forEach((chip) => {
      chip.addEventListener('click', () => {
        if (sendButton.disabled) return;
        input.value = chip.dataset.ask;
        handleMessage();
      });
    });

    document.getElementById('chatBackdrop').addEventListener('click', closePanel);
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape' && panel.classList.contains('active')) closePanel();
    });
  }

  // Only on large screens with a mouse. It re-checks when the window or device mode changes.
  function initCursor() {
    const query = window.matchMedia('(hover: hover) and (pointer: fine) and (min-width: 1025px)');
    const dot = document.getElementById('cursor-dot');
    const ring = document.getElementById('cursor-ring');
    const INTERACTIVE = 'a, button, input, textarea, select, label, .project-showcase, .nav-item-wrapper, .chatbot-button';
    let mouseX = -100;
    let mouseY = -100;
    let ringX = -100;
    let ringY = -100;

    let enabled = false;
    let frame = null;

    const animateRing = () => {
      if (!enabled) {
        frame = null;
        return;
      }
      ringX += (mouseX - ringX) * 0.18;
      ringY += (mouseY - ringY) * 0.18;
      ring.style.transform = `translate(${ringX}px, ${ringY}px) translate(-50%, -50%)`;
      frame = requestAnimationFrame(animateRing);
    };

    const setEnabled = (on) => {
      enabled = on;
      document.body.classList.toggle('custom-cursor', on);
      if (on && !frame) frame = requestAnimationFrame(animateRing);
    };

    query.addEventListener('change', () => setEnabled(query.matches));
    window.addEventListener('pointerdown', (event) => {
      if (event.pointerType !== 'mouse') setEnabled(false);
    }, { passive: true });
    window.addEventListener('pointermove', (event) => {
      if (event.pointerType === 'mouse' && query.matches && !enabled) setEnabled(true);
    }, { passive: true });
    setEnabled(query.matches);

    window.addEventListener('mousemove', (event) => {
      mouseX = event.clientX;
      mouseY = event.clientY;
      dot.style.transform = `translate(${mouseX}px, ${mouseY}px) translate(-50%, -50%)`;
    }, { passive: true });

    // Native video controls need the system cursor, so the custom one hides over them.
    document.addEventListener('mouseover', (event) => {
      const overVideo = Boolean(event.target.closest('.video-container.playing video'));
      const interactive = Boolean(event.target.closest(INTERACTIVE));
      document.body.classList.toggle('cursor-over-video', overVideo);
      ring.classList.toggle('hovered', interactive && !overVideo);
      dot.style.opacity = interactive ? '0' : '1';
    });
  }

  function initClock() {
    const clock = document.getElementById('live-clock');
    const formatter = new Intl.DateTimeFormat('en-US', {
      timeZone: 'Asia/Kolkata',
      hour: '2-digit',
      minute: '2-digit',
      hour12: true
    });
    const update = () => {
      clock.textContent = formatter.format(new Date());
    };
    update();
    setInterval(update, 15000);
  }
});
