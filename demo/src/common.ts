$('.upload-file-btn').on('change', (event) => {
    const target = (event.target || window.event.srcElement) as any;
    const files = target.files;

    if (!FileReader || !files) {
        alert('Your browser does not support file upload. Upgrade your browser.');
        return;
    }

    if (files.length) {
        const reader = new FileReader();
        reader.onload = function () {
            (document.getElementById('target-image') as any).src = reader.result;
        }
        reader.readAsDataURL(files[0]);
    }
});
