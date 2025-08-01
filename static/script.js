document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Element References ---
    const fileInput = document.getElementById('mri-file-input');
    const analyzeBtn = document.getElementById('analyze-btn');
    const fileLabel = document.querySelector('.file-label');
    const loader = document.getElementById('loader');
    const resultsSection = document.getElementById('results-section');
    const errorSection = document.getElementById('error-section');
    const originalImageElem = document.getElementById('original-image');
    const predictionTextElem = document.getElementById('prediction-text').querySelector('span');
    const confidenceTextElem = document.getElementById('confidence-text').querySelector('span');
    const canvas = document.getElementById('3d-canvas');

    let fileHandle = null;

    // --- Three.js Scene Setup ---
    let scene, camera, renderer, controls, brainModel, tumorSphere;
    const textureLoader = new THREE.TextureLoader();

    function init3DScene() {
        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x2a2a2a);

        // --- Camera ---
        const container = document.getElementById('viewer-3d');
        camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
        camera.position.z = 5;

        // --- Renderer ---
        renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
        renderer.setSize(container.clientWidth, container.clientHeight);
        renderer.setPixelRatio(window.devicePixelRatio);

        // --- Lighting ---
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(5, 10, 7.5);
        scene.add(directionalLight);

        // --- Controls ---
        controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;

        // --- Load 3D Brain Model ---
        const loader = new THREE.GLTFLoader();
        // This is a placeholder URL for a generic brain model.
        // You can find free GLB/GLTF models on sites like Sketchfab.
        loader.load('https://cdn.jsdelivr.net/gh/mrdoob/three.js/examples/models/gltf/BrainStem.glb', (gltf) => {
            brainModel = gltf.scene;
            brainModel.scale.set(15, 15, 15); // Adjust scale as needed
            // Ensure the model has a material that can accept a texture
            brainModel.traverse((node) => {
                if (node.isMesh) {
                    node.material = new THREE.MeshStandardMaterial({
                        color: 0xcccccc,
                        metalness: 0.2,
                        roughness: 0.8,
                    });
                }
            });
            scene.add(brainModel);
        }, undefined, (error) => {
            console.error('An error happened while loading the brain model:', error);
        });
        
        // --- Tumor Sphere ---
        const sphereGeometry = new THREE.SphereGeometry(0.5, 32, 32);
        const sphereMaterial = new THREE.MeshStandardMaterial({ color: 0xff0000, visible: false });
        tumorSphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
        tumorSphere.position.x = 3; // Position it to the side of the brain
        scene.add(tumorSphere);


        // --- Animation Loop ---
        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }
        animate();

        // --- Handle Resize ---
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
            fileLabel.textContent = 'Choose an Image';
            analyzeBtn.disabled = true;
        }
    });

    analyzeBtn.addEventListener('click', async () => {
        if (!fileHandle) return;
        
        resultsSection.classList.add('hidden');
        errorSection.classList.add('hidden');
        loader.classList.remove('hidden');

        const reader = new FileReader();
        reader.readAsDataURL(fileHandle);
        reader.onload = async () => {
            const base64StringWithData = reader.result;
            const base64String = base64StringWithData.split(',')[1];
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ file: base64String }),
                });

                if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                
                const data = await response.json();
                displayResults(data, base64StringWithData);

            } catch (error) {
                console.error('Error:', error);
                showError();
            } finally {
                loader.classList.add('hidden');
            }
        };
    });

    // --- Display Logic ---
    function displayResults(data, originalImageBase64) {
        // Update 2D analysis details
        originalImageElem.src = originalImageBase64;
        predictionTextElem.textContent = data.classification;
        confidenceTextElem.textContent = data.confidence;

        // Update 3D visualization
        if (brainModel) {
            // Create a texture from the returned mask
            const maskTexture = textureLoader.load(`data:image/png;base64,${data.mask}`);
            
            // Project the mask onto the brain model
            brainModel.traverse((node) => {
                if (node.isMesh) {
                    // Create a new material that combines the base color with the mask
                    node.material = new THREE.MeshStandardMaterial({
                        color: 0xcccccc,
                        map: node.material.map, // Keep original texture if any
                        alphaMap: maskTexture, // Use the mask for transparency/blending
                        alphaTest: 0.5,
                        transparent: true,
                    });
                }
            });
        }
        
        // Update tumor sphere size and visibility
        if (data.tumor_size > 0.001) { // Only show if tumor is detected
            // Scale the sphere based on the square root of the area ratio
            const scale = Math.sqrt(data.tumor_size) * 5 + 0.1; 
            tumorSphere.scale.set(scale, scale, scale);
            tumorSphere.material.visible = true;
        } else {
            tumorSphere.material.visible = false;
        }

        resultsSection.classList.remove('hidden');
    }

    function showError() {
        errorSection.classList.remove('hidden');
    }

    // --- Initialize ---
    init3DScene();
});
