document.addEventListener('DOMContentLoaded', function() {
    // Mobile navigation toggle
    const navbarToggle = document.querySelector('.navbar-toggle');
    const navbarMenu = document.querySelector('.navbar-menu');
    
    if (navbarToggle && navbarMenu) {
        navbarToggle.addEventListener('click', function() {
            navbarToggle.classList.toggle('active');
            navbarMenu.classList.toggle('active');
        });
    }

    // API Key & Demo Mode Modal
    const openApiModalBtn = document.getElementById('open-api-modal-btn');
    const closeApiModalBtn = document.getElementById('close-api-modal-btn');
    const apiModalBackdrop = document.getElementById('api-modal-backdrop');

    if (openApiModalBtn && apiModalBackdrop) {
        openApiModalBtn.addEventListener('click', function() {
            apiModalBackdrop.classList.add('open');
        });
    }

    if (closeApiModalBtn && apiModalBackdrop) {
        closeApiModalBtn.addEventListener('click', function() {
            apiModalBackdrop.classList.remove('open');
        });
    }

    if (apiModalBackdrop) {
        apiModalBackdrop.addEventListener('click', function(e) {
            if (e.target === apiModalBackdrop) {
                apiModalBackdrop.classList.remove('open');
            }
        });
    }
    
    // Close alert messages
    const closeButtons = document.querySelectorAll('.alert .close-btn');
    
    closeButtons.forEach(button => {
        button.addEventListener('click', function() {
            const alert = this.parentElement;
            alert.style.opacity = '0';
            setTimeout(() => {
                alert.style.display = 'none';
            }, 300);
        });
    });
    
    // Add current year to footer copyright
    const currentYearSpan = document.getElementById('current-year');
    
    if (currentYearSpan) {
        currentYearSpan.textContent = new Date().getFullYear();
    }
    
    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            
            if (targetId === '#') return;
            
            const targetElement = document.querySelector(targetId);
            
            if (targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop - 80,
                    behavior: 'smooth'
                });
                
                // Close mobile menu if open
                if (navbarToggle && navbarMenu && navbarMenu.classList.contains('active')) {
                    navbarToggle.classList.remove('active');
                    navbarMenu.classList.remove('active');
                }
            }
        });
    });
    
    // Add active class to current nav item
    const currentLocation = window.location.pathname;
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        const linkPath = link.getAttribute('href');
        
        if (linkPath === currentLocation || 
            (currentLocation === '/' && linkPath === '/') ||
            (linkPath !== '/' && currentLocation.includes(linkPath))) {
            link.parentElement.classList.add('active');
        }
    });
});
