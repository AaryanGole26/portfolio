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
    const icon = document.getElementById('themeIcon');

    const syncIcon = () => {
      const isLight = document.body.classList.contains('light-mode');
      icon.classList.toggle('bx-moon', isLight);
      icon.classList.toggle('bx-sun', !isLight);
    };

    syncIcon();
    toggle.addEventListener('click', () => {
      const isLight = document.body.classList.toggle('light-mode');
      try {
        localStorage.setItem('theme', isLight ? 'light' : 'dark');
      } catch {
        // The choice still applies to this visit when storage is blocked.
      }
      syncIcon();
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

    const pauseAll = (except) => {
      containers.forEach((container) => {
        if (container === except) return;
        const video = container.querySelector('video');
        if (!video.paused) video.pause();
      });
    };

    containers.forEach((container) => {
      const video = container.querySelector('video');
      const overlay = container.querySelector('.play-overlay-v2');
      const label = overlay.querySelector('.play-label');
      const defaultLabel = label.textContent;
      const poster = container.querySelector('.showcase-img');

      const showUnavailable = () => {
        container.classList.remove('playing', 'buffering');
        video.controls = false;
        label.textContent = 'Could not load. Tap to retry';
      };

      overlay.addEventListener('click', () => {
        pauseAll(container);
        label.textContent = defaultLabel;
        if (poster) video.poster = poster.currentSrc || poster.src;
        if (!video.getAttribute('src') || video.error) {
          video.src = container.dataset.video;
        }
        video.controls = true;
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

    const visibility = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) return;
        const video = entry.target.querySelector('video');
        if (!video.paused) video.pause();
      });
    }, { threshold: 0.2 });
    containers.forEach((container) => visibility.observe(container));

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
        const cardTop = top;
        top += card.offsetHeight;
        return cardTop;
      });
    };

    // A card taller than the screen scrolls to its bottom edge before it sticks.
    const updateStickyOffsets = () => {
      visibleCards.forEach((card) => {
        const offset = Math.min(0, window.innerHeight - card.offsetHeight);
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
      if (isStackedLayout()) videos.pauseAll(activeCard.querySelector('.video-container'));
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
      const navOffset = isStackedLayout() ? 0 : nav.offsetHeight;
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
      button.style.display = 'none';
      input.focus();
    };

    const closePanel = () => {
      panel.classList.remove('active');
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
      const pending = addMessage('Thinking...', 'bot');
      pending.wrapper.classList.add('pending');
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

    document.querySelectorAll('.quick-btn').forEach((quickButton) => {
      quickButton.addEventListener('click', () => {
        addMessage(`Taking you to ${quickButton.textContent}...`, 'bot');
        navigateTo(quickButton.dataset.section);
      });
    });
  }

  function initCursor() {
    const supportsCustomCursor = window.matchMedia('(hover: hover) and (pointer: fine)').matches;
    if (!supportsCustomCursor) return;

    const dot = document.getElementById('cursor-dot');
    const ring = document.getElementById('cursor-ring');
    const INTERACTIVE = 'a, button, input, textarea, select, label, .project-showcase, .nav-item-wrapper, .chatbot-button';
    let mouseX = -100;
    let mouseY = -100;
    let ringX = -100;
    let ringY = -100;

    document.body.classList.add('custom-cursor');

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

    const animateRing = () => {
      ringX += (mouseX - ringX) * 0.18;
      ringY += (mouseY - ringY) * 0.18;
      ring.style.transform = `translate(${ringX}px, ${ringY}px) translate(-50%, -50%)`;
      requestAnimationFrame(animateRing);
    };
    animateRing();
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
