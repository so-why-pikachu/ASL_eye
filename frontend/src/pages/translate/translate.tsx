import { useState, useEffect, useRef } from 'react';
import Webcam from 'react-webcam';
import ModelSequencePlayer from '../../shared/components/ModelSequencePlayer.tsx';
import '../../styles/translate.css';

const API_BASE_URL = '';

export default function Translate() {
    const [ isStreaming, setIsStreaming ] = useState( false );
    const [ isProcessing, setIsProcessing ] = useState( false );
    const [ statusText, setStatusText ] = useState( "Camera is off. Ready to start." );
    const [ translatedText, setTranslatedText ] = useState( "" );
    const [ confidence, setConfidence ] = useState( 0 );

    const [ glbFrames, setGlbFrames ] = useState<string[]>( [] );

    // 🔥 新增：控制播放帧率的 State，默认 5 帧/秒
    const [ playbackFps, setPlaybackFps ] = useState<number>( 5 );

    const [ searchWord, setSearchWord ] = useState( "" );

    const webcamRef = useRef<Webcam>( null );
    const mediaRecorderRef = useRef<MediaRecorder | null>( null );
    const chunksRef = useRef<BlobPart[]>( [] );

    const processAndFetchModel = async ( videoBlob: Blob ) => {
        setIsProcessing( true );
        setStatusText( "Translating video..." );

        const formData = new FormData();
        formData.append( 'video', videoBlob, 'sign_clip.webm' );

        try {
            const predictResponse = await fetch( `${API_BASE_URL}/api/sign/predict`, {
                method: 'POST',
                body: formData,
            } );
            const predictResult = await predictResponse.json();

            if ( predictResult.code === 200 ) {
                const wordName = predictResult.data.word_name;
                setTranslatedText( wordName );
                setConfidence( Math.round( predictResult.data.confidence * 100 ) );
                setStatusText( `Fetching 3D model for "${wordName}"...` );

                const downloadsResponse = await fetch( `${API_BASE_URL}/api/sign/downloads?name=${wordName}` );
                const downloadsResult = await downloadsResponse.json();

                if ( downloadsResult.code === 200 && downloadsResult.data.urls.length > 0 ) {
                    setGlbFrames( downloadsResult.data.urls );
                    setStatusText( "Playback ready." );
                } else {
                    setStatusText( `No 3D model found for "${wordName}".` );
                }
            } else {
                console.warn( "Prediction Error:", predictResult.message );
                setStatusText( "Translation failed: " + predictResult.message );
            }
        } catch ( error ) {
            console.error( "API Error:", error );
            setStatusText( "Network error occurred." );
        } finally {
            setIsProcessing( false );
        }
    };

    useEffect( () => {
        if ( isStreaming ) {
            setGlbFrames( [] );
            setTranslatedText( "" );
            setConfidence( 0 );
            setStatusText( "Camera turning on..." );
            chunksRef.current = [];
        } else {
            if ( mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording' ) {
                mediaRecorderRef.current.stop();
                setStatusText( "Stopping recording..." );
            }
        }
    }, [ isStreaming ] );

    const handleUserMedia = ( stream: MediaStream ) => {
        if ( !isStreaming ) return;

        setStatusText( "Recording..." );
        const mimeType = MediaRecorder.isTypeSupported( 'video/webm' ) ? 'video/webm' : 'video/mp4';
        const mediaRecorder = new MediaRecorder( stream, { mimeType } );

        mediaRecorder.ondataavailable = ( e ) => {
            if ( e.data.size > 0 ) chunksRef.current.push( e.data );
        };

        mediaRecorder.onstop = () => {
            const blob = new Blob( chunksRef.current, { type: mimeType } );
            processAndFetchModel( blob );
        };

        mediaRecorder.start();
        mediaRecorderRef.current = mediaRecorder;
    };

    const fetchModelByName = async ( wordName: string ) => {
        try {
            const downloadsResponse = await fetch( `${API_BASE_URL}/api/sign/downloads?name=${wordName}` );
            const downloadsResult = await downloadsResponse.json();

            if ( downloadsResult.code === 200 && downloadsResult.data.urls.length > 0 ) {
                setGlbFrames( downloadsResult.data.urls );
                setStatusText( "Playback ready." );
            } else {
                setStatusText( `No 3D model found for "${wordName}".` );
                setGlbFrames( [] );
            }
        } catch ( error ) {
            console.error( "API Error:", error );
            setStatusText( "Failed to fetch 3D model." );
        }
    };

    // 🔥 新增：处理文本搜索框提交事件
    const handleTextSearch = async ( e: React.FormEvent ) => {
        e.preventDefault(); // 阻止表单默认刷新
        const word = searchWord.trim().toUpperCase();
        if ( !word ) return;

        setIsProcessing( true );
        setStatusText( `Searching for "${word}"...` );
        setTranslatedText( word );
        setConfidence( 0 ); // 文本搜索不需要置信度
        setGlbFrames( [] );

        await fetchModelByName( word );
        setIsProcessing( false );
    };

    return (
        <div className="sign-container">
            <div className="sign-card">
                {/* 左侧：翻译结果展示 & 3D模型 */}
                <div className="text-section">
                    <div className="text-header">
                        <span className={`live-dot ${isStreaming ? 'recording' : ''}`}></span>
                        {isStreaming ? "RECORDING" : ( isProcessing ? "PROCESSING" : "READY" )}
                    </div>

                    <form
                        onSubmit={handleTextSearch}
                        style={{ display: 'flex', gap: '10px', marginTop: '10px', marginBottom: '15px' }}
                    >
                        <input
                            type="text"
                            value={searchWord}
                            onChange={( e ) => setSearchWord( e.target.value )}
                            placeholder="Type a word (e.g. ACCIDENT)"
                            disabled={isProcessing || isStreaming}
                            style={{
                                flex: 1,
                                padding: '10px 15px',
                                backgroundColor: 'rgba(0, 240, 255, 0.05)',
                                border: '1px solid rgba(0, 240, 255, 0.3)',
                                borderRadius: '8px',
                                color: '#fff',
                                outline: 'none',
                                fontSize: '14px',
                                textTransform: 'uppercase'
                            }}
                        />
                        <button
                            type="submit"
                            disabled={!searchWord.trim() || isProcessing || isStreaming}
                            style={{
                                padding: '10px 20px',
                                backgroundColor: 'rgba(0, 240, 255, 0.15)',
                                border: '1px solid #00f0ff',
                                borderRadius: '8px',
                                color: '#00f0ff',
                                cursor: ( !searchWord.trim() || isProcessing || isStreaming ) ? 'not-allowed' : 'pointer',
                                fontWeight: 'bold',
                                opacity: ( !searchWord.trim() || isProcessing || isStreaming ) ? 0.5 : 1,
                                transition: 'all 0.3s'
                            }}
                        >
                            Search
                        </button>
                    </form>

                    <div className="text-display">
                        <h1 className="main-text">{translatedText || statusText}</h1>
                        {confidence > 0 && (
                            <div className="confidence-badge">
                                Confidence: {confidence}%
                            </div>
                        )}
                    </div>

                    {/* 🔥 把动态的 playbackFps 传给播放器 */}
                    <ModelSequencePlayer
                        isStreaming={isStreaming}
                        isProcessing={isProcessing}
                        glbUrls={glbFrames}
                        fps={playbackFps}
                    />

                    {/* 🔥 新增：FPS 速度控制滑块 (仅在有模型时且没在录制时显示) */}
                    {glbFrames.length > 0 && !isStreaming && (
                        <div className="speed-controller" style={{
                            marginTop: '20px',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '15px',
                            padding: '10px 20px',
                            backgroundColor: 'rgba(0, 240, 255, 0.05)',
                            border: '1px solid rgba(0, 240, 255, 0.2)',
                            borderRadius: '12px'
                        }}>
                            <span style={{
                                color: '#00f0ff',
                                fontSize: '14px',
                                fontWeight: 'bold',
                                minWidth: '100px',
                                textShadow: '0 0 5px rgba(0, 240, 255, 0.5)'
                            }}>
                                SPEED: {playbackFps} FPS
                            </span>
                            <input
                                type="range"
                                min="1"
                                max="15"
                                value={playbackFps}
                                onChange={( e ) => setPlaybackFps( Number( e.target.value ) )}
                                style={{
                                    flex: 1,
                                    accentColor: '#00f0ff',
                                    cursor: 'pointer'
                                }}
                            />
                        </div>
                    )}

                    <div className="instruction">
                        <p>Click "Start Camera", perform your gesture, then click "Stop & Translate".</p>
                    </div>
                </div>

                {/* 右侧：摄像头输入 */}
                <div className="camera-section">
                    <div className="camera-wrapper">
                        {isStreaming ? (
                            <>
                                <Webcam
                                    ref={webcamRef}
                                    audio={false}
                                    className="webcam-video"
                                    videoConstraints={{ facingMode: "user" }}
                                    onUserMedia={handleUserMedia}
                                />
                                <div className="skeleton-overlay"><HandSkeletonSVG /></div>
                            </>
                        ) : (
                            <div className="camera-placeholder">
                                <div className="placeholder-icon">📷</div>
                            </div>
                        )}
                    </div>

                    <div className="control-bar">
                        <button
                            className={`toggle-btn ${isStreaming ? 'active' : ''}`}
                            onClick={() => setIsStreaming( !isStreaming )}
                            disabled={isProcessing}
                        >
                            {isStreaming ? "Stop & Translate" : "Start Camera"}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}

// SVG Component
const HandSkeletonSVG = () => (
    <svg viewBox="0 0 200 200" className="hand-svg">
        <circle cx="100" cy="150" r="4" fill="#00f0ff" />
        <path d="M100 150 L60 140 L40 120 L30 100" stroke="rgba(0, 240, 255, 0.6)" strokeWidth="2" fill="none" />
        <circle cx="60" cy="140" r="3" fill="#00f0ff" />
        <circle cx="40" cy="120" r="3" fill="#00f0ff" />
        <circle cx="30" cy="100" r="3" fill="#00f0ff" />
        <path d="M100 150 L90 110 L85 80 L80 50" stroke="rgba(0, 240, 255, 0.6)" strokeWidth="2" fill="none" />
        <circle cx="90" cy="110" r="3" fill="#00f0ff" />
        <circle cx="85" cy="80" r="3" fill="#00f0ff" />
        <circle cx="80" cy="50" r="3" fill="#00f0ff" />
        <path d="M100 150 L105 105 L108 70 L110 40" stroke="rgba(0, 240, 255, 0.6)" strokeWidth="2" fill="none" />
        <circle cx="105" cy="105" r="3" fill="#00f0ff" />
        <circle cx="108" cy="70" r="3" fill="#00f0ff" />
        <circle cx="110" cy="40" r="3" fill="#00f0ff" />
        <circle cx="100" cy="150" r="10" stroke="#00f0ff" strokeWidth="1" fill="none" className="pulse-circle" />
    </svg>
);