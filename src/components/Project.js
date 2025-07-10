import React, { useState, useRef } from 'react';
import './Project.css';
import PointCloudViewer from './PointCloudViewer';

import GitHubIcon from '@mui/icons-material/GitHub';
import ArrowBackIosNewIcon from '@mui/icons-material/ArrowBackIosNew';
import ArrowForwardIosIcon from '@mui/icons-material/ArrowForwardIos';
import Button from '@mui/material/Button';

const Project = () => {
    const [currentPointCloud, setCurrentPointCloud] = useState(0);
    const pointCloudsRef = useRef(null);
    const [currentApplicabilityVideo, setCurrentApplicabilityVideo] = useState(0);
    const applicabilityVideosRef = useRef(null);
    const [currentDataEfficiencyContent, setCurrentDataEfficiencyContent] = useState(0);
    const dataEfficiencyContentsRef = useRef(null);
    const [currentRealTimeVideo, setCurrentRealTimeVideo] = useState(0);
    const realTimeVideosRef = useRef(null);
    const [isHoveringPointCloud, setIsHoveringPointCloud] = useState(false);

    const scrollToCurrentPointCloud = (index) => {
        const container = pointCloudsRef.current;
        const pointClouds = container.querySelectorAll('.project__methodDataCollectionPointCloud');
        if (pointClouds[index]) {
            pointClouds[index].scrollIntoView({
                behavior: 'smooth',
                inline: 'center',
                block: 'nearest',
            });
        }
    };

    const onClickPointCloudNavigatorButton = (index) => {
        return () => {
            setCurrentPointCloud(index);
            scrollToCurrentPointCloud(index);
        }
    };

    const onHoverPointCloud = () => {
        setIsHoveringPointCloud(true);
    }

    const onLeavePointCloud = () => {
        setIsHoveringPointCloud(false);
    }

    const scrollToCurrentApplicabilityVideo = (index) => {
        const container = applicabilityVideosRef.current;
        const applicabilityVideos = container.querySelectorAll('.project__applicabilityVideo');
        if (applicabilityVideos[index]) {
            applicabilityVideos[index].scrollIntoView({
                behavior: 'smooth',
                inline: 'center',
                block: 'nearest',
            });
        }
    };

    const incrementApplicabilityVideo = () => {
        if (currentApplicabilityVideo < 3) {
            const next = currentApplicabilityVideo + 1;
            setCurrentApplicabilityVideo(next);
            scrollToCurrentApplicabilityVideo(next);
        }
    };

    const decrementApplicabilityVideo = () => {
        if (currentApplicabilityVideo > 0) {
            const prev = currentApplicabilityVideo - 1;
            setCurrentApplicabilityVideo(prev);
            scrollToCurrentApplicabilityVideo(prev);
        }
    };

    const scrollToCurrentDataEfficiencyContent = (index) => {
        const container = dataEfficiencyContentsRef.current;
        const dataEfficiencyContents = container.querySelectorAll('.project__dataEfficiencyBody');
        if (dataEfficiencyContents[index]) {
            dataEfficiencyContents[index].scrollIntoView({
                behavior: 'smooth',
                inline: 'center',
                block: 'nearest',
            });
        }
    };

    const onClickDataEfficiencyNavigatorButton = (index) => {
        return () => {
            setCurrentDataEfficiencyContent(index);
            scrollToCurrentDataEfficiencyContent(index);
        }
    };

    const incrementDataEfficiencyContent = () => {
        if (currentDataEfficiencyContent < 4) {
            const next = currentDataEfficiencyContent + 1;
            setCurrentDataEfficiencyContent(next);
            scrollToCurrentDataEfficiencyContent(next);
        }
    };

    const decrementDataEfficiencyContent = () => {
        if (currentDataEfficiencyContent > 0) {
            const prev = currentDataEfficiencyContent - 1;
            setCurrentDataEfficiencyContent(prev);
            scrollToCurrentDataEfficiencyContent(prev);
        }
    };

    const scrollToCurrentRealTimeVideo = (index) => {
        const container = realTimeVideosRef.current;
        const realTimeVideos = container.querySelectorAll('.project__realtimeVideo');
        if (realTimeVideos[index]) {
            realTimeVideos[index].scrollIntoView({
                behavior: 'smooth',
                inline: 'center',
                block: 'nearest',
            });
        }
    };

    const incrementRealTimeVideo = () => {
        if (currentRealTimeVideo < 3) {
            const next = currentRealTimeVideo + 1;
            setCurrentRealTimeVideo(next);
            scrollToCurrentRealTimeVideo(next);
        }
    };

    const decrementRealTimeVideo = () => {
        if (currentRealTimeVideo > 0) {
            const prev = currentRealTimeVideo - 1;
            setCurrentRealTimeVideo(prev);
            scrollToCurrentRealTimeVideo(prev);
        }
    };

    return (
        <div className="project">
            <div className="project__header">
                <video className="project__headerVideo" loop muted autoPlay playsInline>
                    <source src="./videos/Teaser_x8_1k_web.mov" type="video/mp4" />
                    Your browser does not support the video tag.
                </video>
                <div className="project__headerTitle">
                    <p className="project__headerMainTitle google-sans-semibold white-color">N2M</p>
                    <p className="project__headerSubTitle google-sans-regular white-color">Bridging Navigation and Manipulation by Learning Initial Pose Preference from Rollout</p>
                </div>
                <div className="project__headerNavigator">
                    <a href="#overview" className="project__headerNavigatorItem">
                        <p className="project__headerNavigatorItemTitle google-sans-regular white-color">Overview</p>
                    </a>
                    <a href="#method" className="project__headerNavigatorItem">
                        <p className="project__headerNavigatorItemTitle google-sans-regular white-color">Method</p>
                    </a>
                    <a href="#key-features" className="project__headerNavigatorItem">
                        <p className="project__headerNavigatorItemTitle google-sans-regular white-color">Key Features</p>
                    </a>
                    <a href="#future-work" className="project__headerNavigatorItem">
                        <p className="project__headerNavigatorItemTitle google-sans-regular white-color">Future Work</p>
                    </a>
                </div>
            </div>
            <div className="project__additionalInfo">
                <div className="project__authors">
                    <p className="project__author google-sans-regular blue-color"><a href="https://cckaixin.github.io/myWebsite/" target="_blank" rel="noopener noreferrer">Kaixin Chai</a><sup className="black-color">&dagger;1,2</sup></p>
                    <p className="project__author google-sans-regular blue-color"><a href="https://hjl1013.github.io/" target="_blank" rel="noopener noreferrer">Hyunjun Lee</a><sup className="black-color">&dagger;1,3</sup></p>
                    <p className="project__author google-sans-regular blue-color"><a href="https://jeongjun-kim.github.io/" target="_blank" rel="noopener noreferrer">Jeongjun Kim</a><sup className="black-color">1,3</sup></p>
                    <p className="project__author google-sans-regular blue-color"><a href="https://www.linkedin.com/in/sunwoo-kim-299493201/" target="_blank" rel="noopener noreferrer">Sunwoo Kim</a><sup className="black-color">1,3</sup></p>
                    <p className="project__author google-sans-regular blue-color"><a href="https://sites.google.com/view/junseunglee/home/about" target="_blank" rel="noopener noreferrer">Junseung Lee</a><sup className="black-color">1,3</sup></p>
                    <p className="project__author google-sans-regular blue-color"><a href="https://dhleekr.github.io/" target="_blank" rel="noopener noreferrer">Doohyun Lee</a><sup className="black-color">1</sup></p>
                    <p className="project__author google-sans-regular blue-color"><a href="https://minoring.github.io/" target="_blank" rel="noopener noreferrer">Minho Heo</a><sup className="black-color">1</sup></p>
                    <p className="project__author google-sans-regular blue-color"><a href="https://clvrai.com/web_lim/" target="_blank" rel="noopener noreferrer">Joseph J. Lim</a><sup className="black-color">&Dagger;1</sup></p>
                </div>
                <div className="project__affiliations">
                    <p className="project__affiliation google-sans-regular"><sup className="black-color">1</sup>KAIST</p>
                    <p className="project__affiliation google-sans-regular"><sup className="black-color">2</sup>Xi’an Jiaotong University</p>
                    <p className="project__affiliation google-sans-regular"><sup className="black-color">3</sup>Seoul National University</p>
                </div>
                <div className="project__symbols">
                    <p className="project__symbolsText google-sans-regular">&dagger; equal contribution</p>
                    <p className="project__symbolsText google-sans-regular">&Dagger; corresponding author</p>
                </div>
                <div className="project__materials">
                    <a href="https://arxiv.org/abs/2507.03303" target="_blank" rel="noopener noreferrer">
                        <div className="project__material dark-gray-background">
                            <img src='./icons/arxiv.png' alt="paper" className="project__materialIcon" />
                            <p className="project__materialName google-sans-regular white-color">arXiv</p>
                        </div>
                    </a>
                    <a href="https://arxiv.org/abs/2507.03303" target="_blank" rel="noopener noreferrer">
                        <div className="project__material dark-gray-background">
                            <GitHubIcon className="project__materialIcon white-color" />
                            <p className="project__materialName google-sans-regular white-color">Code</p>
                        </div>
                    </a>
                    <a href="https://arxiv.org/abs/2507.03303" target="_blank" rel="noopener noreferrer">
                        <div className="project__material dark-gray-background">
                            <img src='./icons/youtube.png' alt="youtube" className="project__materialIcon" />
                            <p className="project__materialName google-sans-regular white-color">Youtube</p>
                        </div>
                    </a>
                    <a href="https://arxiv.org/abs/2507.03303" target="_blank" rel="noopener noreferrer">
                        <div className="project__material dark-gray-background">
                            <img src='./icons/bilibili.png' alt="bilibili" className="project__materialIcon" />
                            <p className="project__materialName google-sans-regular white-color">Bilibili</p>
                        </div>
                    </a>
                </div>
            </div>


            <div className="project__body">
                <div className="project__bodyContent project__overview" id="overview">
                    <p className="project__bodyContentTitle project__overview google-sans-semibold">Overview</p>
                    <video className="project__overviewVideo" loop muted autoPlay controls playsInline>
                        <source src="./videos/Overview_2k.mov" type="video/mp4" />
                        Your browser does not support the video tag.
                    </video>
                </div>

                <div className="project__bodyContentWrapper light-gray-background">
                    <div className="project__bodyContent project__abstract">
                        <p className="project__bodyContentTitle google-sans-semibold">Abstract</p>
                        <p className="project__abstractText google-sans-regular">
                        In mobile manipulation, the manipulation policy has strong preferences for initial poses where it is executed. 
                        However, the navigation module focuses solely on reaching the task area, without considering which initial pose is preferable for downstream manipulation.
                        To address this misalignment, we present N2M, a transition module that guides the robot to a preferable initial pose after reaching the task area, thereby substantially improving task success rates. 
                        N2M features several key advantages: (1) reliance solely on ego-centric observation without requiring global or historical information; 
                        (2) real-time adaptation to environmental changes; 
                        (3) accurate predictions with high viewpoint robustness; 
                        (4) broad applicability across diverse tasks, manipulation policies and robot hardware; and 
                        (5) remarkable data efficiency and generalizability.
                        We demonstrate the effectiveness of N2M through extensive simulation and real-world experiments. In the PnPCounterToCab task, N2M improves success rates from 3% with reachability-based baselines to 54%, 
                        and in the Toy Box Handover task, with only 15 data samples collected in one room, N2M can give reliable predictions throughout the whole room and even generalize to entirely unseen environments.
                        </p>
                    </div>
                </div>

                <div className="project__bodyContent project__method" id="method">
                    <p className="project__bodyContentTitle google-sans-semibold">Method</p>
                    <div className="project__methodContent">
                        <p className="project__methodContentTitle google-sans-semibold">Pipeline</p>
                        <img src='./figures/Method_System_Overview.png' alt="method_system_overview" className="project__methodPipelineImage" />
                        <p className="project__methodContentText google-sans-regular">Navigate to task area &rarr; <span className="blue-color google-sans-semibold">Adjust pose with N2M</span> &rarr; Execute manipulation policy</p>
                    </div>
                    <div className="project__methodContent project__methodDataCollection">
                        <p className="project__methodContentTitle google-sans-semibold">Data Collection</p>
                        <div className="project__methodDataCollectionBody">
                            <img src='./figures/Method_Data_Preparation.png' alt="method_data_preparation" className="project__methodDataCollectionImage" />
                            <div className="project__methodDataCollectionPointCloudsWrapper" onMouseEnter={onHoverPointCloud} onMouseLeave={onLeavePointCloud}>
                                <div className="project__methodDataCollectionPointCloudsNavigator">
                                    <div className={`project__methodDataCollectionPointCloudsNavigatorButton light-gray-background ${currentPointCloud === 0 ? 'project__methodDataCollectionPointCloudsNavigatorButton--selected' : ''}`} onClick={onClickPointCloudNavigatorButton(0)}><p className="project__methodDataCollectionPointCloudsNavigatorButtonText google-sans-regular">Raw</p></div>
                                    <div className={`project__methodDataCollectionPointCloudsNavigatorButton light-gray-background ${currentPointCloud === 1 ? 'project__methodDataCollectionPointCloudsNavigatorButton--selected' : ''}`} onClick={onClickPointCloudNavigatorButton(1)}><p className="project__methodDataCollectionPointCloudsNavigatorButtonText google-sans-regular">Augmented 1</p></div>
                                    <div className={`project__methodDataCollectionPointCloudsNavigatorButton light-gray-background ${currentPointCloud === 2 ? 'project__methodDataCollectionPointCloudsNavigatorButton--selected' : ''}`} onClick={onClickPointCloudNavigatorButton(2)}><p className="project__methodDataCollectionPointCloudsNavigatorButtonText google-sans-regular">Augmented 2</p></div>
                                    <div className={`project__methodDataCollectionPointCloudsNavigatorButton light-gray-background ${currentPointCloud === 3 ? 'project__methodDataCollectionPointCloudsNavigatorButton--selected' : ''}`} onClick={onClickPointCloudNavigatorButton(3)}><p className="project__methodDataCollectionPointCloudsNavigatorButtonText google-sans-regular">Augmented 3</p></div>
                                </div>
                                <div className={`project__methodDataCollection3DHelperWrapper ${isHoveringPointCloud ? '' : 'project__methodDataCollection3DHelperWrapper--visible'}`}><img src='./icons/3d_helper.png' alt="3d_helper" className="project__methodDataCollection3DHelper" /></div>
                                <div className="project__methodDataCollectionPointClouds" ref={pointCloudsRef}>
                                    <div className="project__methodDataCollectionPointCloud">
                                        <PointCloudViewer pcdPath='./pcls/local_scene.pcd' />
                                    </div>
                                    <div className="project__methodDataCollectionPointCloud">
                                        <PointCloudViewer pcdPath='./pcls/rendered1.pcd' />
                                    </div>
                                    <div className="project__methodDataCollectionPointCloud">
                                        <PointCloudViewer pcdPath='./pcls/rendered2.pcd' />
                                    </div>
                                    <div className="project__methodDataCollectionPointCloud">
                                        <PointCloudViewer pcdPath='./pcls/rendered3.pcd' />
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div className="project__bodyContent project__keyFeatures" id="key-features">
                    <p className="project__bodyContentTitle google-sans-semibold">Key Features</p>
                    <div className="project__keyFeature">
                        <p className="project__keyFeatureTitle google-sans-semibold"><span className="blue-color google-sans-semibold">Ego-centric</span> Prediction</p>
                        <div className="project__keyFeatureBody project__egoCentricBody">
                            <img src='./figures/Key_Feature_Ego-centric.png' alt="key_feature_ego-centric" className="project__egoCentricImage" />
                            <div className="project__egoCentricTextBody">
                                <p className="project__egoCentricText google-sans-regular">
                                    ✅ N2M only relies on ego-centric observations without requiring global or historical information
                                </p>
                                <p className="project__egoCentricText google-sans-regular">
                                    ❌ Doesn’t require global / historical information such as pre-built map
                                </p>
                            </div>
                        </div>
                    </div>

                    <div className="project__keyFeature">
                        <p className="project__keyFeatureTitle google-sans-semibold"><span className="blue-color google-sans-semibold">Real-time</span> Inference</p>
                        <div className="project_realTimeVideosWrapper">
                            <div className="project__realTimeVideosLeftArrow project__realTimeVideosArrow dark-gray-background" onClick={decrementRealTimeVideo}>
                                <ArrowBackIosNewIcon className="white-color" />
                            </div>
                            <div className="project__realTimeVideosRightArrow project__realTimeVideosArrow dark-gray-background" onClick={incrementRealTimeVideo}>
                                <ArrowForwardIosIcon className="white-color" />
                            </div>
                            <div className="project__realTimeVideos" ref={realTimeVideosRef}>
                                <video className="project__realtimeVideo" loop muted autoPlay playsInline>
                                    <source src="./videos/Realtime_pushchair_x8_05k_web.mov" type="video/mp4" />
                                    Your browser does not support the video tag.
                                </video>
                                <video className="project__realtimeVideo" loop muted autoPlay playsInline>
                                    <source src="./videos/Realtime_laptop_x8_05k_web.mov" type="video/mp4" />
                                    Your browser does not support the video tag.
                                </video>
                                <video className="project__realtimeVideo" loop muted autoPlay playsInline>
                                    <source src="./videos/Realtime_microwave_x8_5k_web.mov" type="video/mp4" />
                                    Your browser does not support the video tag.
                                </video>
                                <video className="project__realtimeVideo" loop muted autoPlay playsInline>
                                    <source src="./videos/Realtime_TBH_x8_05k_web.mov" type="video/mp4" />
                                    Your browser does not support the video tag.
                                </video>
                            </div>
                        </div>
                        <p className="project__realTimeText google-sans-regular">Environment is typically non-static. N2M can adjust its prediction real-time.</p>
                    </div>

                    <div className="project__keyFeature">
                        <p className="project__keyFeatureTitle google-sans-semibold"><span className="blue-color google-sans-semibold">Viewpoint</span> Robustness</p>
                        <div className="project__keyFeatureBody project__viewpointRobustnessBody">
                            <video className="project__viewpointRobustnessVideo" loop muted autoPlay playsInline>
                                <source src="./videos/Robustness_lamp_x8_05k_web.mov" type="video/mp4" />
                                Your browser does not support the video tag.
                            </video>
                        </div>
                        <p className="project__viewpointRobustnessText google-sans-regular">End pose of navigation can be anywhere within the task area. N2M is able to give reliable prediction at different poses.</p>
                    </div>

                    <div className="project__keyFeature">
                        <p className="project__keyFeatureTitle google-sans-semibold">Broad <span className="blue-color google-sans-semibold">applicability</span></p>
                        <div className="project__applicabilityMedias">
                            <div className="project__applicabilityVideoWrapper">
                                <div className="project__applicabilityVideosLeftArrow project__applicabilityVideosArrow dark-gray-background" onClick={decrementApplicabilityVideo}>
                                    <ArrowBackIosNewIcon className="white-color" />
                                </div>
                                <div className="project__applicabilityVideosRightArrow project__applicabilityVideosArrow dark-gray-background" onClick={incrementApplicabilityVideo}>
                                    <ArrowForwardIosIcon className="white-color" />
                                </div>
                                <div className="project__applicabilityVideos" ref={applicabilityVideosRef}>
                                    <video className="project__applicabilityVideo" loop muted autoPlay playsInline>
                                        <source src="./videos/Applicability_pnp_x8_05k_web.mov" type="video/mp4" />
                                        Your browser does not support the video tag.
                                    </video>
                                    <video className="project__applicabilityVideo" loop muted autoPlay playsInline>
                                        <source src="./videos/Applicability_closedoubledoors_x8_05k_web.mov" type="video/mp4" />
                                        Your browser does not support the video tag.
                                    </video>
                                    <video className="project__applicabilityVideo" loop muted autoPlay playsInline>
                                        <source src="./videos/Applicability_opensingledoor_x8_05k_web.mov" type="video/mp4" />
                                        Your browser does not support the video tag.
                                    </video>
                                    <video className="project__applicabilityVideo" loop muted autoPlay playsInline>
                                        <source src="./videos/Applicability_closedrawer_x8_05k_web.mov" type="video/mp4" />
                                        Your browser does not support the video tag.
                                    </video>
                                </div>
                            </div>
                            <img src='./figures/Broad_applicability.png' alt="key_feature_applicability" className="project__applicabilityImage" />
                        </div>
                        <p className="project__applicabilityText google-sans-regular">
                            N2M doesn’t rely on any assumptions about tasks, policies, and hardware, which makes it widely applicable <br/>
                            Through quantitative experiments, we demonstrate that N2M can significantly improve the success rate across various tasks and manipulation policies. <br/>
                            <span className="project__applicabilityTextFootnote">*Reachability baseline: Reachability based naive integration between navigation and manipulation</span>
                        </p>
                    </div>

                    <div className="project__keyFeature">
                        <p className="project__keyFeatureTitle google-sans-semibold"><span className="blue-color google-sans-semibold">Data efficiency</span> and <span className="blue-color google-sans-semibold">generalizability</span></p>
                        <div className="project__keyFeatureBody project__dataEfficiencyBodyWrapper">
                            <div className="project__dataEfficiencyNavigator">
                                <div className={`project__dataEfficiencyNavigatorButton project__dataEfficiencyNavigatorButtonFirstThree light-gray-background ${currentDataEfficiencyContent === 0 ? 'project__dataEfficiencyNavigatorButton--selected' : ''}`} onClick={onClickDataEfficiencyNavigatorButton(0)}><p className="project__methodDataCollectionPointCloudsNavigatorButtonText google-sans-regular">Data efficiency study</p></div>
                                <div className={`project__dataEfficiencyNavigatorButton project__dataEfficiencyNavigatorButtonFirstThree light-gray-background ${currentDataEfficiencyContent === 1 ? 'project__dataEfficiencyNavigatorButton--selected' : ''}`} onClick={onClickDataEfficiencyNavigatorButton(1)}><p className="project__methodDataCollectionPointCloudsNavigatorButtonText google-sans-regular">Generalizability study 1</p></div>
                                <div className={`project__dataEfficiencyNavigatorButton project__dataEfficiencyNavigatorButtonFirstThree light-gray-background ${currentDataEfficiencyContent === 2 ? 'project__dataEfficiencyNavigatorButton--selected' : ''}`} onClick={onClickDataEfficiencyNavigatorButton(2)}><p className="project__methodDataCollectionPointCloudsNavigatorButtonText google-sans-regular">Generalizability study 2</p></div>
                                <div className={`project__dataEfficiencyNavigatorButton project__dataEfficiencyNavigatorButtonLastTwo light-gray-background ${currentDataEfficiencyContent === 3 ? 'project__dataEfficiencyNavigatorButton--selected' : ''}`} onClick={onClickDataEfficiencyNavigatorButton(3)}><p className="project__methodDataCollectionPointCloudsNavigatorButtonText google-sans-regular">Comprehensive test case 1</p></div>
                                <div className={`project__dataEfficiencyNavigatorButton project__dataEfficiencyNavigatorButtonLastTwo light-gray-background ${currentDataEfficiencyContent === 4 ? 'project__dataEfficiencyNavigatorButton--selected' : ''}`} onClick={onClickDataEfficiencyNavigatorButton(4)}><p className="project__methodDataCollectionPointCloudsNavigatorButtonText google-sans-regular">Comprehensive test case 2</p></div>
                            </div>
                            <div className="project__dataEfficiencyBodyContents" ref={dataEfficiencyContentsRef}>
                                <div className="project__dataEfficiencyBody project__dataEfficiencyExp3a">
                                    <div className="project__dataEfficiencyExp3aMedias project__dataEfficiencyMedias">
                                        <video className="project__dataEfficiencyExp3aVideo" loop muted autoPlay playsInline>
                                            <source src="./videos/Applicability_pnp_x8_05k_web.mov" type="video/mp4" />
                                            Your browser does not support the video tag.
                                        </video>
                                        <img src='./figures/Data_Efficiency_Exp3a_graph.png' alt="data_efficiency_exp3a" className="project__dataEfficiencyExp3aImage" />
                                    </div>
                                    <p className="project__dataEfficiencyText project__dataEfficiencyExp3aText google-sans-regular">
                                        With just around 10 data, N2M learns the initial pose preference of apple pick and place task.
                                    </p>
                                </div>
                                <div className="project__dataEfficiencyBody project__dataEfficiencyExp3b">
                                    <div className="project__dataEfficiencyExp3bMedias project__dataEfficiencyMedias">
                                        <img src='./figures/Data_Efficiency_Exp3b_scenes.png' alt="data_efficiency_exp3b_scenes" className="project__dataEfficiencyExp3bScenesImage" />
                                        <img src='./figures/Data_Efficiency_Exp3b_graph.png' alt="data_efficiency_exp3b_graph" className="project__dataEfficiencyExp3bGraphImage" />
                                    </div>
                                    <p className="project__dataEfficiencyText project__dataEfficiencyExp3bText google-sans-regular">
                                        For the same task, even when the furniture <span className="blue-color google-sans-semibold">texture</span> in the test scene is unseen during training, N2M still learns preferences and generalizes effectively.
                                    </p>
                                </div>
                                <div className="project__dataEfficiencyBody project__dataEfficiencyExp3c">
                                    <div className="project__dataEfficiencyExp3cMedias project__dataEfficiencyMedias">
                                        <img src='./figures/Data_Efficiency_Exp3c_scenes.png' alt="data_efficiency_exp3c_scenes" className="project__dataEfficiencyExp3cScenesImage" />
                                        <img src='./figures/Data_Efficiency_Exp3c_graph.png' alt="data_efficiency_exp3c_graph" className="project__dataEfficiencyExp3cGraphImage" />
                                    </div>
                                    <p className="project__dataEfficiencyText project__dataEfficiencyExp3cText google-sans-regular">
                                        For the same task, even when the furniture <span className="blue-color google-sans-semibold">layout</span> in the test scene is unseen during training, N2M still learns preferences and generalizes effectively.
                                    </p>
                                </div>
                                <div className="project__dataEfficiencyBody project__dataEfficiencyExp4">
                                    <div className="project__dataEfficiencyExp4Medias project__dataEfficiencyMedias">
                                        <img src='./figures/Data_Efficiency_Exp4.png' alt="data_efficiency_exp4" className="project__dataEfficiencyExp4Image" />
                                    </div>
                                    <div className="project__dataEfficiencyExp4TextBody">
                                        <p className="project__dataEfficiencyText project__dataEfficiencyExp4TextMain google-sans-semibold">The robot attempts to retrieve a lamp from the shelf. The 3×4 table shows the success rate for each cell. <br/>We collect one rollout from each cell marked in <span className="blue-color">blue</span> for N2M training.</p>
                                        <div className="project__dataEfficiencyExp4TextResults">
                                            <p className="project__dataEfficiencyText project__dataEfficiencyExp4TextResult google-sans-regular"><span className="blue-color google-sans-semibold">Result (a)</span>: Simply ensuring reachability is not enough to determine a preferable initial pose.</p>
                                            <p className="project__dataEfficiencyText project__dataEfficiencyExp4TextResult google-sans-regular"><span className="blue-color google-sans-semibold">Result (b) & (c)</span>: Even with rollouts collected from only a subset of cells, N2M can still generate reasonable predictions for unseen cells. <br/></p>
                                            <p className="project__dataEfficiencyText project__dataEfficiencyExp4TextResult google-sans-regular"><span className="blue-color google-sans-semibold">Result (d)</span>: With rollouts from all 12 cells, N2M achieves the highest success rate. For cells where the success rate is relatively low, we observe that the bottleneck lies in the manipulation policy while N2M is able to give reasonable predictions.</p>
                                        </div>
                                    </div>
                                </div>
                                <div className="project__dataEfficiencyBody project__dataEfficiencyExp5">
                                    <div className="project__dataEfficiencyExp5Medias project__dataEfficiencyMedias">
                                        <video className="project__dataEfficiencyExp5Video" loop muted autoPlay playsInline>
                                            <source src="./videos/Generalizability_all_in_one_x8_05k_web.mov" type="video/mp4" />
                                            Your browser does not support the video tag.
                                        </video>
                                    </div>
                                    <div className="project__dataEfficiencyExp5TextBody">
                                        <p className="project__dataEfficiencyText project__dataEfficiencyExp5TextMain google-sans-semibold">The robot attempts to take the toy box from human. We collected <span className="blue-color google-sans-semibold">15 data samples</span> to make N2M learn the initial pose preference for this task.</p>
                                        <p className="project__dataEfficiencyText project__dataEfficiencyExp5TextExplanation google-sans-regular">Despite all test scenes being entirely different from the training environment, N2M generalizes well and gives reasonable initial pose prediction.</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div className="project__bodyContentWrapper light-gray-background">
                    <div className="project__bodyContent project__futureWork" id="future-work">
                        <p className="project__bodyContentTitle google-sans-semibold">Future Work</p>
                        <div className="project__futureWorkBody">
                            <p className="project__futureWorkText google-sans-regular">🚀 <span className="google-sans-semibold">Hardware dependency</span>: N2M relies on high quality depth estimation to capture realistic point cloud. Enabling N2M to run with only an RGB camera through monocular depth estimation and scene reconstruction to reduce hardware dependencies.</p>
                            <p className="project__futureWorkText google-sans-regular">🚀 <span className="google-sans-semibold">Incorporating failure rollouts</span>: N2M only learns from positive rollouts. This can lead to overestimation failing to avoid failure initial poses. Learning from failure rollouts can prevent overestimation of initial pose preference and help the module find poses where robot can achieve higher success rate</p>
                        </div>
                    </div>
                </div>

                <div className="project__bodyContent project__bibtex">
                    <p className="project__bodyContentTitle google-sans-semibold">Bibtex</p>
                    <div className="project__bibtexBody">
                        <pre className="project__bibtexText google-sans-regular light-gray-background">
                            TBD
                        </pre>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Project;