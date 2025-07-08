import React, { useState, useRef } from 'react';
import './Project.css';
import PointCloudViewer from './PointCloudViewer';

import GitHubIcon from '@mui/icons-material/GitHub';
import ArrowBackIosNewIcon from '@mui/icons-material/ArrowBackIosNew';
import ArrowForwardIosIcon from '@mui/icons-material/ArrowForwardIos';

const Project = () => {
    const [currentPointCloud, setCurrentPointCloud] = useState(0);
    const pointCloudsRef = useRef(null);

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

    const incrementPointCloud = () => {
        if (currentPointCloud < 3) {
            const next = currentPointCloud + 1;
            setCurrentPointCloud(next);
            scrollToCurrentPointCloud(next);
        }
    };

    const decrementPointCloud = () => {
        if (currentPointCloud > 0) {
            const prev = currentPointCloud - 1;
            setCurrentPointCloud(prev);
            scrollToCurrentPointCloud(prev);
        }
    };

    return (
        <div className="project">
            <div className="project__header">
                <video className="project__headerVideo" loop muted autoPlay>
                    <source src="./videos/Teaser_x8_05k_web.mov" type="video/mp4" />
                    Your browser does not support the video tag.
                </video>
                <div className="project__headerTitle">
                    <p className="project__headerMainTitle google-sans-semibold white-color">N2M</p>
                    <p className="project__headerSubTitle google-sans-regular white-color">Bridging Navigation and Manipulation by Learning <br/>Initial Pose Preference from Rollout</p>
                </div>
                <div className="project__headerNavigator">
                    <div className="project__headerNavigatorItem">
                        <p className="project__headerNavigatorItemTitle google-sans-regular white-color">Overview</p>
                    </div>
                    <div className="project__headerNavigatorItem">
                        <p className="project__headerNavigatorItemTitle google-sans-regular white-color">Key Features</p>
                    </div>
                    <div className="project__headerNavigatorItem">
                        <p className="project__headerNavigatorItemTitle google-sans-regular white-color">Failure Cases</p>
                    </div>
                    <div className="project__headerNavigatorItem">
                        <p className="project__headerNavigatorItemTitle google-sans-regular white-color">Future Work</p>
                    </div>
                </div>
            </div>
            <div className="project__additionalInfo">
                <div className="project__authors">
                    <div className="project__authorsOneLine">
                        <p className="project__author google-sans-regular blue-color"><a href="https://cckaixin.github.io/myWebsite/" target="_blank" rel="noopener noreferrer">Kaixin Chai</a><sup className="black-color">&dagger;1,2</sup></p>
                        <p className="project__author google-sans-regular blue-color"><a href="https://hjl1013.github.io/" target="_blank" rel="noopener noreferrer">Hyunjun Lee</a><sup className="black-color">&dagger;1,3</sup></p>
                        <p className="project__author google-sans-regular blue-color"><a href="https://jeongjun-kim.github.io/" target="_blank" rel="noopener noreferrer">Jeongjun Kim</a><sup className="black-color">1,3</sup></p>
                    </div>
                    <div className="project__authorsOneLine">
                        <p className="project__author google-sans-regular blue-color"><a href="https://www.linkedin.com/in/sunwoo-kim-299493201/" target="_blank" rel="noopener noreferrer">Sunwoo Kim</a><sup className="black-color">1,3</sup></p>
                        <p className="project__author google-sans-regular blue-color"><a href="https://sites.google.com/view/junseunglee/home/about" target="_blank" rel="noopener noreferrer">Junseung Lee</a><sup className="black-color">1,3</sup></p>
                        <p className="project__author google-sans-regular blue-color"><a href="https://dhleekr.github.io/" target="_blank" rel="noopener noreferrer">Doohyun Lee</a><sup className="black-color">1</sup></p>
                        <p className="project__author google-sans-regular blue-color"><a href="https://minoring.github.io/" target="_blank" rel="noopener noreferrer">Minho Heo</a><sup className="black-color">1</sup></p>
                        <p className="project__author google-sans-regular blue-color"><a href="https://clvrai.com/web_lim/" target="_blank" rel="noopener noreferrer">Joseph J. Lim</a><sup className="black-color">1</sup></p>
                    </div>
                </div>
                <div className="project__affiliations">
                    <p className="project__affiliation google-sans-regular"><sup className="black-color">1</sup>KAIST</p>
                    <p className="project__affiliation google-sans-regular"><sup className="black-color">2</sup>Xi’an Jiaotong University</p>
                    <p className="project__affiliation google-sans-regular"><sup className="black-color">3</sup>Seoul National University</p>
                </div>
                <div className="project__materials">
                    <a href="https://arxiv.org/abs/2507.03303" target="_blank" rel="noopener noreferrer">
                        <div className="project__material dark-gray-background">
                            <img src="./icons/arxiv.png" alt="paper" className="project__materialIcon" />
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
                            <img src="./icons/youtube.png" alt="youtube" className="project__materialIcon" />
                            <p className="project__materialName google-sans-regular white-color">Youtube</p>
                        </div>
                    </a>
                    <a href="https://arxiv.org/abs/2507.03303" target="_blank" rel="noopener noreferrer">
                        <div className="project__material dark-gray-background">
                            <img src="./icons/bilibili.png" alt="bilibili" className="project__materialIcon" />
                            <p className="project__materialName google-sans-regular white-color">Bilibili</p>
                        </div>
                    </a>
                </div>
            </div>


            <div className="project__body">
                <div className="project__bodyContent project__overview">
                    <p className="project__bodyContentTitle project__overview google-sans-semibold">Overview</p>
                    <video className="project__overviewVideo" loop muted autoPlay controls>
                        <source src="./videos/RA-L overview.mov" type="video/mp4" />
                        Your browser does not support the video tag.
                    </video>
                </div>

                <div className="project__bodyContentWrapper light-gray-background">
                    <div className="project__bodyContent project__abstract">
                        <p className="project__bodyContentTitle google-sans-semibold">Abstract</p>
                        <p className="project__abstractText google-sans-regular">
                        In mobile manipulation, robots first navigate to a task area and then execute pre-trained manipulation policies to complete the task. 
                        Due to hardware constraints, environmental factors, and the distribution of training data, the initial pose from which a manipulation policy is executed has a significant impact on the task success rate. 
                        However, navigation modules typically fail to account for these pose preferences. To address this critical gap, we present N2M, a transition module that guides the robot to a preferable initial pose after reaching the task area, thereby substantially improving task success rates. 
                        N2M features several key advantages: (1) real-time adaptation to environmental changes; (2) reliance solely on onboard RGB-D camera without requiring global or historical information; 
                        (3) accurate predictions with high viewpoint robustness; (4) remarkable data efficiency and generalizability; and (5) applicability across diverse tasks, manipulation policies and robot hardware configurations. 
                        Through extensive experiments in both simulation and real-world settings, we demonstrate the effectiveness of our method. 
                        In the PnPCounterToCab task, N2M improves success rates from 3% with reachability-based baselines to 54%, and in the Toy Box Handover task, with only 15 data samples collected in one room, N2M can give reliable predictions throughout the whole room and even generalize to entirely unseen environments.
                        </p>
                    </div>
                </div>

                <div className="project__bodyContent project__method">
                    <p className="project__bodyContentTitle google-sans-semibold">Method</p>
                    <div className="project__methodContent">
                        <p className="project__methodContentTitle google-sans-semibold">Pipeline</p>
                        <img src="./figures/Method_System_Overview.png" alt="method_system_overview" className="project__methodPipelineImage" />
                        <p className="project__methodContentText google-sans-regular">Navigate within task area &rarr; <span className="blue-color google-sans-semibold">Adjust pose with N2M</span> &rarr; Execute manipulation policy</p>
                    </div>
                    <div className="project__methodContent project__methodDataCollection">
                        <p className="project__methodContentTitle google-sans-semibold">Data Collection</p>
                        <div className="project__methodDataCollectionBody">
                            <img src="./figures/Method_Data_Preparation.png" alt="method_data_preparation" className="project__methodDataCollectionImage" />
                            <div className="project__methodDataCollectionPointCloudsWrapper">
                                <div className="project__methodDataCollectionPointCloudsLeftArrow project__methodDataCollectionPointCloudsArrow dark-gray-background" onClick={decrementPointCloud}>
                                    <ArrowBackIosNewIcon className="white-color" />
                                </div>
                                <div className="project__methodDataCollectionPointCloudsRightArrow project__methodDataCollectionPointCloudsArrow dark-gray-background" onClick={incrementPointCloud}>
                                    <ArrowForwardIosIcon className="white-color" />
                                </div>
                                <div className="project__methodDataCollectionPointClouds" ref={pointCloudsRef}>
                                    <div className="project__methodDataCollectionPointCloud">
                                        <PointCloudViewer pcdPath="./pcls/local_scene.pcd" />
                                        <p className="project__methodDataCollectionPointCloudTitle google-sans-regular">Local Scene</p>
                                    </div>
                                    <div className="project__methodDataCollectionPointCloud">
                                        <PointCloudViewer pcdPath="./pcls/rendered1.pcd" />
                                        <p className="project__methodDataCollectionPointCloudTitle google-sans-regular">Rendered 1</p>
                                    </div>
                                    <div className="project__methodDataCollectionPointCloud">
                                        <PointCloudViewer pcdPath="./pcls/rendered2.pcd" />
                                        <p className="project__methodDataCollectionPointCloudTitle google-sans-regular">Rendered 2</p>
                                    </div>
                                    <div className="project__methodDataCollectionPointCloud">
                                        <PointCloudViewer pcdPath="./pcls/rendered3.pcd" />
                                        <p className="project__methodDataCollectionPointCloudTitle google-sans-regular">Rendered 3</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div className="project__bodyContent project__keyFeatures">
                    <p className="project__bodyContentTitle google-sans-semibold">Key Features</p>
                    <div className="project__keyFeature">
                        <p className="project__keyFeatureTitle google-sans-semibold"><span className="blue-color google-sans-semibold">Ego-centric</span> Prediction</p>
                        <div className="project__keyFeatureBody">
                            <img src="./figures/Key_Feature_Ego-centric.png" alt="key_feature_ego-centric" className="project__keyFeatureImage" />
                            <div className="project__keyFeatureTextBody">
                                <p className="project__keyFeatureText google-sans-regular">
                                    ✅ N2M only requires ego-centric observations and produces predictions based on ego-centric coordinate
                                </p>
                                <p className="project__keyFeatureText google-sans-regular">
                                    ❌ Doesn’t require global / historical information such as pre-built map
                                </p>
                            </div>
                        </div>
                    </div>

                    <div className="project__keyFeature">
                        <p className="project__keyFeatureTitle google-sans-semibold"><span className="blue-color google-sans-semibold">Real-time</span> Inference</p>
                        <div className="project__keyFeatureBody">
                            <img src="./figures/Key_Feature_Ego-centric.png" alt="key_feature_ego-centric" className="project__keyFeatureImage" />
                            <div className="project__keyFeatureTextBody">
                                <p className="project__keyFeatureText google-sans-regular">
                                    ✅ N2M only requires ego-centric observations and produces predictions based on ego-centric coordinate
                                </p>
                                <p className="project__keyFeatureText google-sans-regular">
                                    ❌ Doesn’t require global / historical information such as pre-built map
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Project;