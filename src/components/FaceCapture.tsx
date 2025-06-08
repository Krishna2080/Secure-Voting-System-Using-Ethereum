import { useState, useRef, useEffect } from 'react';
import { faceRecognitionApi } from '../services/faceRecognitionApi';

interface FaceCaptureProps {
  onCapture: (result: any) => void;
  onCancel: () => void;
  mode: 'register' | 'authenticate';
  voterData?: { name: string; email: string };
}

const FaceCapture = ({ onCapture, onCancel, mode, voterData }: FaceCaptureProps) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [status, setStatus] = useState('Initializing camera...');
  const [backendStatus, setBackendStatus] = useState<'checking' | 'online' | 'offline'>('checking');

  useEffect(() => {
    checkBackendStatus();
    initializeCamera();
    
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  const checkBackendStatus = async () => {
    try {
      const isHealthy = await faceRecognitionApi.healthCheck();
      setBackendStatus(isHealthy ? 'online' : 'offline');
    } catch (error) {
      setBackendStatus('offline');
    }
  };

  const initializeCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' }
      });
      
      setStream(mediaStream);
      
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        videoRef.current.play();
      }
      
      setStatus('Camera ready - Position your face in the frame');
    } catch (error) {
      console.error('Error accessing camera:', error);
      setStatus('Camera access denied');
    }
  };

  const captureImageAsBase64 = (): string | null => {
    if (!videoRef.current || !canvasRef.current) return null;
    
    const canvas = canvasRef.current;
    const video = videoRef.current;
    const context = canvas.getContext('2d');
    
    if (!context) return null;
    
    // Set canvas dimensions to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Draw current video frame to canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Get base64 image data
    return canvas.toDataURL('image/jpeg', 0.8);
  };

  const captureAndProcess = async () => {
    if (backendStatus !== 'online') {
      setStatus('Backend server is offline. Please start the Python FastAPI server.');
      return;
    }

    setIsProcessing(true);
    
    if (mode === 'register') {
      setStatus('Processing registration with fraud prevention checks...');
    } else {
      setStatus('Processing authentication...');
    }
    
    try {
      const imageData = captureImageAsBase64();
      
      if (!imageData) {
        throw new Error('Failed to capture image');
      }

      let result;
      
      if (mode === 'register' && voterData) {
        result = await faceRecognitionApi.registerVoter({
          name: voterData.name,
          email: voterData.email,
          image_data: imageData
        });
      } else if (mode === 'authenticate') {
        result = await faceRecognitionApi.authenticateVoter({
          image_data: imageData
        });
      }

      if (result?.success) {
        if (mode === 'register') {
          setStatus('Registration successful with fraud prevention verification!');
        } else {
          setStatus('Authentication successful!');
        }
        onCapture(result);
      } else {
        const errorMessage = result?.message || `${mode === 'register' ? 'Registration' : 'Authentication'} failed`;
        setStatus(errorMessage);
        
        // Keep error message visible longer for fraud prevention alerts
        if (errorMessage.includes('already registered under the name')) {
          setTimeout(() => setStatus('Ready to try again'), 5000);
        } else {
          setTimeout(() => setStatus('Ready to try again'), 3000);
        }
      }
      
    } catch (error) {
      console.error(`Error during ${mode}:`, error);
      setStatus(`Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
      setTimeout(() => setStatus('Ready to try again'), 3000);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="bg-gray-50 rounded-lg p-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Camera Feed */}
        <div className="space-y-4">
          <div className="relative bg-black rounded-lg overflow-hidden aspect-video">
            <video
              ref={videoRef}
              className="w-full h-full object-cover"
              autoPlay
              muted
              playsInline
            />
            
            {/* Face detection overlay */}
            <div className="absolute inset-0 border-2 border-blue-500 border-dashed m-8 rounded-lg flex items-center justify-center">
              <div className="text-white text-center bg-black bg-opacity-50 px-4 py-2 rounded">
                <p className="text-sm">Position your face here</p>
              </div>
            </div>
          </div>
          
          <div className="text-center">
            {/* Backend Status */}
            <div className="mb-3">
              <span className={`inline-flex items-center px-2 py-1 text-xs rounded-full ${
                backendStatus === 'online' ? 'bg-green-100 text-green-800' :
                backendStatus === 'offline' ? 'bg-red-100 text-red-800' :
                'bg-yellow-100 text-yellow-800'
              }`}>
                <span className={`w-2 h-2 rounded-full mr-1 ${
                  backendStatus === 'online' ? 'bg-green-400' :
                  backendStatus === 'offline' ? 'bg-red-400' :
                  'bg-yellow-400'
                }`}></span>
                Backend {backendStatus}
              </span>
            </div>

            <p className="text-sm text-gray-600 mb-3">{status}</p>
            <div className="flex justify-center space-x-3">
              <button
                onClick={captureAndProcess}
                disabled={isProcessing || !stream || backendStatus !== 'online'}
                className="px-6 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
              >
                {isProcessing ? 'Processing...' : 
                 mode === 'register' ? 'Capture & Verify' : 'Authenticate'
                }
              </button>
              <button
                onClick={onCancel}
                className="px-6 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
        
        {/* Instructions */}
        <div className="space-y-6">
          <div>
            <h4 className="text-lg font-medium text-gray-900 mb-3">
              {mode === 'register' ? 'Biometric Registration' : 'Face Authentication'} Instructions
            </h4>
            <ul className="space-y-2 text-sm text-gray-600">
              <li className="flex items-start">
                <span className="flex-shrink-0 w-5 h-5 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center text-xs font-bold mr-3 mt-0.5">1</span>
                <span>Position your face within the blue frame</span>
              </li>
              <li className="flex items-start">
                <span className="flex-shrink-0 w-5 h-5 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center text-xs font-bold mr-3 mt-0.5">2</span>
                <span>Look directly at the camera</span>
              </li>
              <li className="flex items-start">
                <span className="flex-shrink-0 w-5 h-5 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center text-xs font-bold mr-3 mt-0.5">3</span>
                <span>Ensure good lighting on your face</span>
              </li>
              <li className="flex items-start">
                <span className="flex-shrink-0 w-5 h-5 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center text-xs font-bold mr-3 mt-0.5">4</span>
                <span>Click the capture button when ready</span>
              </li>
            </ul>
          </div>
          
          {mode === 'register' && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <div className="flex">
                <svg className="w-5 h-5 text-red-400 mr-3 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                </svg>
                <div>
                  <h5 className="text-sm font-medium text-red-900">Fraud Prevention Active</h5>
                  <p className="text-sm text-red-700 mt-1">
                    System will check if your face is already registered under a different name. Each person can only register once.
                  </p>
                </div>
              </div>
            </div>
          )}
          
          

          {backendStatus === 'offline' && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <div className="flex">
                <svg className="w-5 h-5 text-red-400 mr-3 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
                <div>
                  <h5 className="text-sm font-medium text-red-900">Backend Required</h5>
                  <p className="text-sm text-red-700 mt-1">
                    Please run the Python FastAPI server: <code>python backend/main.py</code>
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
      
      {/* Hidden canvas for image processing */}
      <canvas ref={canvasRef} className="hidden" />
    </div>
  );
};

export default FaceCapture;
