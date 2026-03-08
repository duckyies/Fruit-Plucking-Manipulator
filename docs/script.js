// ===============================
// PAGE INTERACTION LOGIC
// ===============================

// Enhanced smooth scrolling with visual feedback
document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
  anchor.addEventListener("click", function (e) {
    e.preventDefault();
    const target = document.querySelector(this.getAttribute("href"));
    if (target) {
      target.scrollIntoView({ behavior: "smooth", block: "start" });
      // Add subtle visual feedback
      target.style.animation = "pulse 0.6s ease-out";
    }
  });
});

// Add animation styles dynamically
const animationStyles = document.createElement('style');
animationStyles.textContent = `
  @keyframes pulse {
    0% { opacity: 1; }
    100% { opacity: 1; }
  }
  
  @keyframes fadeInUp {
    from {
      opacity: 0;
      transform: translateY(20px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }
  
  @keyframes slideIn {
    from {
      opacity: 0;
      max-height: 0;
    }
    to {
      opacity: 1;
      max-height: 1000px;
    }
  }
`;
document.head.appendChild(animationStyles);

// Enhanced expandable timeline with smooth transitions
const timelineItems = document.querySelectorAll(".timeline-item");
timelineItems.forEach((item, index) => {
  item.addEventListener("click", function(e) {
    e.stopPropagation();
    const content = this.querySelector(".timeline-content");
    if (content) {
      const isVisible = content.style.display === "block";
      
      // Close other timeline items with animation
      timelineItems.forEach((otherItem, otherIndex) => {
        if (otherItem !== this) {
          const otherContent = otherItem.querySelector(".timeline-content");
          if (otherContent && otherContent.style.display === "block") {
            otherContent.style.animation = "none";
            otherContent.style.opacity = "0";
            otherContent.style.maxHeight = "0";
            otherContent.style.overflow = "hidden";
            otherContent.style.transition = "all 0.3s ease-out";
            setTimeout(() => otherContent.style.display = "none", 300);
          }
        }
      });
      
      // Toggle current item with smooth animation
      if (isVisible) {
        content.style.opacity = "0";
        content.style.maxHeight = "0";
        setTimeout(() => content.style.display = "none", 300);
      } else {
        content.style.display = "block";
        content.style.opacity = "1";
        content.style.maxHeight = "1000px";
        content.style.transition = "all 0.3s ease-out";
        content.style.animation = "slideIn 0.4s ease-out";
      }
    }
  });
});

// Close timeline items on document click
document.addEventListener('click', () => {
  timelineItems.forEach((item) => {
    const content = item.querySelector(".timeline-content");
    if (content && content.style.display === "block") {
      content.style.display = "none";
    }
  });
});

// Enhanced header scroll styling with smooth transitions
const header = document.querySelector(".site-header");
let lastScroll = 0;
let scrollDirection = 'down';

window.addEventListener("scroll", () => {
  const currentScroll = window.scrollY;
  
  if (currentScroll > 50) {
    header.style.background = "rgba(18, 18, 25, 0.9)";
    header.style.boxShadow = "0 4px 20px rgba(165, 107, 255, 0.1)";
    header.style.backdropFilter = "blur(20px)";
  } else {
    header.style.background = "rgba(18, 18, 25, 0.55)";
    header.style.boxShadow = "none";
    header.style.backdropFilter = "blur(14px)";
  }
  
  lastScroll = currentScroll;
});

// Intersection Observer for elegant scroll reveal animations
const revealOptions = {
  threshold: 0.1,
  rootMargin: '0px 0px -50px 0px'
};

const revealOnScroll = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      // Apply reveal animation
      entry.target.style.animation = "fadeInUp 0.6s cubic-bezier(0.34, 1.56, 0.64, 1) forwards";
      // Optional: stop observing after revealing
      revealOnScroll.unobserve(entry.target);
    }
  });
}, revealOptions);

// Observe cards, subsections, and images for scroll reveal
document.querySelectorAll('.card, .subsection, .method-image, .sim-preview').forEach((el) => {
  el.style.opacity = '0';
  el.style.transform = 'translateY(20px)';
  revealOnScroll.observe(el);
});

// Add interactive effects to cards
document.querySelectorAll('.card').forEach(card => {
  card.addEventListener('mouseenter', function() {
    this.style.transform = 'translateY(-8px) scale(1.02)';
  });
  
  card.addEventListener('mouseleave', function() {
    this.style.transform = 'translateY(0) scale(1)';
  });
});

// Smooth scroll performance optimization
let ticking = false;

window.addEventListener('scroll', () => {
  if (!ticking) {
    window.requestAnimationFrame(() => {
      // Update header on scroll
      ticking = false;
    });
    ticking = true;
  }
});

// Page load animation
window.addEventListener('load', () => {
  document.body.style.opacity = '1';
});

// ===============================
// FRUIT DETECTION SIMULATION
// ===============================

function simulateDetection(label) {
  const el = document.getElementById("prediction");
  if (el) {
    const color = label === "ripe" ? "#e8b84b" : "#5dbf5d";
    el.innerHTML = `<strong>Detected:</strong> <span style="color:${color};">${label.toUpperCase()} BANANA</span>`;
    el.style.animation = "fadeInUp 0.3s ease-out";
  }
  moveRobot(label === "ripe" ? "red" : "green");
}

// ===============================
// ROBOT MOTION
// ===============================

function moveRobot(color) {
  let target = color === "green" ? greenCube.position : redCube.position;

  let step = 0;

  let motion = setInterval(() => {
    step += 0.02;

    let angle = Math.atan2(target.x, target.z);
    j1.rotation.y = angle;
    j2.rotation.z = -step;
    j3.rotation.z = step * 0.7;

    if (step > 1) {
      clearInterval(motion);
      pickObject(color);
    }
  }, 30);
}

function pickObject(color) {
  let cube = color === "green" ? greenCube : redCube;

  let height = 0;

  let lift = setInterval(() => {
    height += 0.03;

    cube.position.y = 0.3 + height;

    if (height > 1.5) {
      clearInterval(lift);

      setTimeout(() => {
        placeObject(cube);
      }, 5000);
    }
  }, 30);
}

function placeObject(cube) {
  let height = 1.5;

  let down = setInterval(() => {
    height -= 0.03;

    cube.position.y = 0.3 + height;

    if (height <= 0) {
      clearInterval(down);
      resetRobot();
    }
  }, 30);
}

function resetRobot() {
  let step = 1;

  let motion = setInterval(() => {
    step += 0.02;

    j2.rotation.z = -0.8 * step;
    j3.rotation.z = 0.5 * step;

    if (step > 1) {
      clearInterval(motion);
      pickObject(color);
    }
  }, 30);
}

// ===============================
// THREE.JS ROBOT SIMULATION
// ===============================

const container = document.getElementById("robot-sim");

if (!container) {
  console.warn("robot-sim container not found");
} else {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0f172a);

  const camera = new THREE.PerspectiveCamera(
    60,
    container.clientWidth / container.clientHeight,
    0.1,
    100,
  );

  camera.position.set(4, 3, 6);
  camera.lookAt(0, 1, 0);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(container.clientWidth, container.clientHeight);

  container.appendChild(renderer.domElement);

  // Orbit controls
  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;

  // ===== LIGHTING =====

  // soft global light
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
  scene.add(ambientLight);

  // main directional light (like the sun)
  const light1 = new THREE.DirectionalLight(0xffffff, 1);
  light1.position.set(5, 10, 5);
  scene.add(light1);

  // secondary fill light for shadows
  const light2 = new THREE.DirectionalLight(0x88aaff, 0.6);
  light2.position.set(-5, 4, -4);
  scene.add(light2);
  // floor
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(20, 20),
    new THREE.MeshStandardMaterial({ color: 0x222222 }),
  );

  floor.rotation.x = -Math.PI / 2;
  scene.add(floor);

  const grid = new THREE.GridHelper(20, 20, 0x444444, 0x222222);
  scene.add(grid);

  // cubes
  const cubeGeo = new THREE.BoxGeometry(0.6, 0.6, 0.6);

  window.redCube = new THREE.Mesh(
    cubeGeo,
    new THREE.MeshStandardMaterial({ color: "red" }),
  );

  window.greenCube = new THREE.Mesh(
    cubeGeo,
    new THREE.MeshStandardMaterial({ color: "green" }),
  );

  redCube.position.set(-2, 0.3, 0);
  greenCube.position.set(2, 0.3, 0);

  scene.add(redCube);
  scene.add(greenCube);

  // robot material
  const mat = new THREE.MeshStandardMaterial({ color: 0x4f7cff });

  // ===============================
  // ROBOT STRUCTURE (6 DOF STYLE)
  // ===============================

  const base = new THREE.Mesh(new THREE.CylinderGeometry(0.5, 0.5, 0.4), mat);

  scene.add(base);

  window.j1 = new THREE.Group();
  j1.position.y = 0.2;
  base.add(j1);

  const link1 = new THREE.Mesh(new THREE.BoxGeometry(0.6, 1.5, 0.6), mat);

  link1.position.y = 1;
  j1.add(link1);

  window.j2 = new THREE.Group();
  j2.position.y = 2;
  j1.add(j2);

  const link2 = new THREE.Mesh(new THREE.BoxGeometry(0.5, 1.4, 0.5), mat);

  link2.position.y = 1;
  j2.add(link2);

  window.j3 = new THREE.Group();
  j3.position.y = 2;
  j2.add(j3);

  const link3 = new THREE.Mesh(new THREE.BoxGeometry(0.3, 1.5, 0.3), mat);

  link3.position.y = 0.75;
  j3.add(link3);

  const j4 = new THREE.Group();
  j4.position.y = 1.5;
  j3.add(j4);

  const link4 = new THREE.Mesh(new THREE.BoxGeometry(0.25, 1, 0.25), mat);

  link4.position.y = 0.5;
  j4.add(link4);

  const j5 = new THREE.Group();
  j5.position.y = 1;
  j4.add(j5);

  const j6 = new THREE.Group();
  j6.position.y = 0.3;
  j5.add(j6);

  const gripper = new THREE.Mesh(
    new THREE.BoxGeometry(0.5, 0.2, 0.5),
    new THREE.MeshStandardMaterial({ color: 0xff4444 }),
  );

  j6.add(gripper);

  // ===============================
  // ANIMATION LOOP
  // ===============================

  function animate() {
    requestAnimationFrame(animate);

    controls.update();

    renderer.render(scene, camera);
  }

  animate();

  // ===============================
  // RESIZE HANDLING
  // ===============================

  window.addEventListener("resize", () => {
    camera.aspect = container.clientWidth / container.clientHeight;

    camera.updateProjectionMatrix();

    renderer.setSize(container.clientWidth, container.clientHeight);
  });
}
