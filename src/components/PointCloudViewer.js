import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { PCDLoader } from 'three/examples/jsm/loaders/PCDLoader';
import { TrackballControls } from 'three/examples/jsm/controls/TrackballControls';

const PointCloudViewer = ({ pcdPath }) => {
    const containerRef = useRef(null);

    useEffect(() => {
        const container = containerRef.current;
        if (!container) return;

        // Set up scene, camera, and renderer
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xF1F1F1);
        const camera = new THREE.PerspectiveCamera(
        75,
        container.clientWidth / container.clientHeight,
        0.1,
        1000
        );
        
        camera.position.set(0, 0, 3); // Set camera position
        camera.up.set(1, 0, 0);

        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(container.clientWidth, container.clientHeight);
        container.appendChild(renderer.domElement);

        // Add OrbitControls for interaction
        const controls = new TrackballControls(camera, renderer.domElement);
        controls.rotateSpeed = 5.0;
        controls.zoomSpeed = 2.0;
        controls.panSpeed = 1.0;

        // Load and add the point cloud
        const loader = new PCDLoader();
        loader.load(pcdPath, (points) => {
        scene.add(points);
        });

        // Animate and render
        const animate = () => {
        requestAnimationFrame(animate);
        controls.update(); // update controls each frame
        renderer.render(scene, camera);
        };
        animate();

        // Cleanup
        return () => {
            renderer.dispose();
            container.innerHTML = '';
        };
    }, [pcdPath]);

    return <div className="project__methodDataCollectionPointCloudCanvas" ref={containerRef} />;
};

export default PointCloudViewer;