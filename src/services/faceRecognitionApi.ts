
const API_BASE_URL = 'http://localhost:8000';

export interface VoterRegistrationRequest {
  name: string;
  email: string;
  image_data: string;
}

export interface FaceAuthenticationRequest {
  image_data: string;
}

export interface VoteRequest {
  voter_name: string;
  candidate_id: string;
}

export interface ApiResponse<T> {
  success: boolean;
  message: string;
  data?: T;
  blockchain_result?: any;
  tx_hash?: string;
  block_number?: number;
  local_backup?: boolean;
}

export const faceRecognitionApi = {
  async registerVoter(data: VoterRegistrationRequest): Promise<ApiResponse<any>> {
    try {
      const response = await fetch(`${API_BASE_URL}/register-voter`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });
      
      const result = await response.json();
      return result;
    } catch (error) {
      console.error('Voter registration failed:', error);
      throw new Error('Failed to register voter');
    }
  },

  async authenticateVoter(data: FaceAuthenticationRequest): Promise<ApiResponse<any>> {
    try {
      const response = await fetch(`${API_BASE_URL}/authenticate-voter`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });
      
      const result = await response.json();
      return result;
    } catch (error) {
      console.error('Voter authentication failed:', error);
      throw new Error('Failed to authenticate voter');
    }
  },

  async castVote(data: VoteRequest): Promise<ApiResponse<any>> {
    try {
      const response = await fetch(`${API_BASE_URL}/cast-vote`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });
      
      const result = await response.json();
      return result;
    } catch (error) {
      console.error('Vote casting failed:', error);
      throw new Error('Failed to cast vote');
    }
  },

  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      const result = await response.json();
      return result.status === 'healthy';
    } catch (error) {
      console.error('Backend health check failed:', error);
      return false;
    }
  }
};
