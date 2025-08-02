document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Element References ---
    const fileInput = document.getElementById('mri-file-input');
    const analyzeBtn = document.getElementById('analyze-btn');
    const fileLabel = document.querySelector('.file-label');
    const loader = document.getElementById('loader');
    const resultsSection = document.getElementById('results-section');
    const errorSection = document.getElementById('error-section');
    
    // 2D Viewer elements
    const sliceImageElem = document.getElementById('slice-image');
    const maskOverlayElem = document.getElementById('mask-overlay');
    const sliceSlider = document.getElementById('slice-slider');
    const sliceNumberElem = document.getElementById('slice-number');
    const totalSlicesElem = document.getElementById('total-slices');

    // 3D Viewer elements
    const canvas = document.getElementById('3d-canvas');

    let fileHandle = null;
    let brainSlices = [];
    let maskSlices = [];

    // --- Three.js Scene Setup ---
    let scene, camera, renderer, controls, tumorMesh;

    function init3DScene() {
        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x2a2a2a);

        const container = document.getElementById('viewer-3d');
        camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
        camera.position.z = 150;

        renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
        renderer.setSize(container.clientWidth, container.clientHeight);
        renderer.setPixelRatio(window.devicePixelRatio);

        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        scene.add(ambientLight);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
        directionalLight.position.set(0, 0, 1);
        scene.add(directionalLight);

        controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;

        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }
        animate();

        window.addEventListener('resize', () => {
            camera.aspect = container.clientWidth / container.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, container.clientHeight);
        });
    }

    // --- Event Listeners ---
    fileInput.addEventListener('change', (e) => {
        fileHandle = e.target.files[0];
        if (fileHandle) {
            fileLabel.textContent = fileHandle.name;
            analyzeBtn.disabled = false;
        } else {
            fileLabel.textContent = 'Choose a NIfTI or .zip File';
            analyzeBtn.disabled = true;
        }
    });

    analyzeBtn.addEventListener('click', async () => {
        if (!fileHandle) return;
        
        resultsSection.classList.add('hidden');
        errorSection.classList.add('hidden');
        loader.classList.remove('hidden');

        const formData = new FormData();
        formData.append('file', fileHandle);

        try {
            const response = await fetch('/predict_3d', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.error || `HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            displayResults(data);

        } catch (error) {
            console.error('Error:', error);
            showError(error.message);
        } finally {
            loader.classList.add('hidden');
        }
    });

    sliceSlider.addEventListener('input', (e) => {
        updateSliceViewer(parseInt(e.target.value));
    });

    function updateSliceViewer(sliceIndex) {
        if (brainSlices.length > 0) {
            sliceImageElem.src = `data:image/png;base64,${brainSlices[sliceIndex]}`;
            maskOverlayElem.src = `data:image/png;base64,${maskSlices[sliceIndex]}`;
            sliceNumberElem.textContent = sliceIndex + 1;
        }
    }

    // --- Display Logic ---
    function displayResults(data) {
        // Clear previous 3D model if it exists
        if (tumorMesh) {
            scene.remove(tumorMesh);
            tumorMesh.geometry.dispose();
            tumorMesh.material.dispose();
        }

        // Update 2D Slice Viewer
        brainSlices = data.brain_slices;
        maskSlices = data.mask_slices;
        sliceSlider.max = brainSlices.length - 1;
        sliceSlider.value = Math.floor(brainSlices.length / 2);
        totalSlicesElem.textContent = brainSlices.length;
        updateSliceViewer(parseInt(sliceSlider.value));

        // Update 3D Tumor Mesh Viewer
        if (data.tumor_mesh) {
            const { vertices, faces } = data.tumor_mesh;
            const geometry = new THREE.BufferGeometry();
            
            const flatVerts = vertices.flat();
            const flatFaces = faces.flat();

            geometry.setAttribute('position', new THREE.Float32BufferAttribute(flatVerts, 3));
            geometry.setIndex(flatFaces);
            geometry.computeVertexNormals();

            const material = new THREE.MeshStandardMaterial({
                color: 0xff0000,
                opacity: 0.6,
                transparent: true,
                side: THREE.DoubleSide
            });

            tumorMesh = new THREE.Mesh(geometry, material);
            
            // Center the mesh
            geometry.center();

            scene.add(tumorMesh);
        }

        resultsSection.classList.remove('hidden');
    }

    function showError(message) {
        errorSection.querySelector('p').textContent = message;
        errorSection.classList.remove('hidden');
    }

    // --- Initialize ---
    init3DScene();
});
