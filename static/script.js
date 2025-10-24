function showModal(category, title, body) {
    const modal = document.getElementById('customModal');
    const modalContent = document.getElementById('modalContent');
    const modalTitle = document.getElementById('modalTitle');
    const modalBody = document.getElementById('modalBody');
    const modalIcon = document.getElementById('modalIcon');

    // Reset classes
    modalContent.className = 'modal-content';
    modalIcon.innerHTML = '';
    
    // Set content and style
    modalTitle.textContent = title;
    modalBody.innerHTML = body;

    if (category === 'danger') {
        modalContent.classList.add('modal-danger');
        modalIcon.innerHTML = '<i class="fas fa-times-circle text-red-500"></i>';
    } else if (category === 'success') {
        modalContent.classList.add('modal-success');
        modalIcon.innerHTML = '<i class="fas fa-check-circle text-green-500"></i>';
    } else {
        // Warning (Sử dụng tạm)
        modalContent.classList.add('modal-warning');
        modalIcon.innerHTML = '<i class="fas fa-exclamation-triangle text-yellow-500"></i>';
    }

    modal.style.display = 'block';
}

function closeModal() {
    document.getElementById('customModal').style.display = 'none';
}

// Đóng modal khi click ra ngoài
window.onclick = function(event) {
    const modal = document.getElementById('customModal');
    if (event.target == modal) {
        modal.style.display = "none";
    }
}
