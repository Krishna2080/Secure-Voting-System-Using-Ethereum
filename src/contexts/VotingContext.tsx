
import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

export interface Candidate {
  id: string;
  name: string;
  party: string;
  votes: number;
}

export interface Voter {
  id: string;
  name: string;
  email: string;
  faceEmbedding: number[];
}

export interface Vote {
  id: string;
  voterName: string;
  candidateId: string;
  timestamp: string;
  txHash?: string;
}

export interface CandidateResult extends Candidate {
  percentage: number;
}

export interface VotingContextType {
  candidates: Candidate[];
  voters: Voter[];
  votes: Vote[];
  currentVoter: Voter | null;
  electionName: string;
  addCandidate: (candidate: Omit<Candidate, "votes">) => void;
  removeCandidate: (id: string) => void;
  vote: (candidateId: string) => void;
  addVoter: (voter: Omit<Voter, "id">) => void;
  setCurrentVoter: (voter: Voter | null) => void;
  castVote: (candidateId: string) => boolean;
  getResults: () => CandidateResult[];
  getTotalVotes: () => number;
  clearAllData: () => void;
  setElectionName: (name: string) => void;
}

const VotingContext = createContext<VotingContextType | undefined>(undefined);

export const useVoting = () => {
  const context = useContext(VotingContext);
  if (!context) {
    throw new Error('useVoting must be used within a VotingProvider');
  }
  return context;
};

interface VotingProviderProps {
  children: ReactNode;
}

export const VotingProvider: React.FC<VotingProviderProps> = ({ children }) => {
  const [candidates, setCandidates] = useState<Candidate[]>([
    { id: '1', name: 'John Smith', party: 'Democratic Party', votes: 0 },
    { id: '2', name: 'Jane Johnson', party: 'Republican Party', votes: 0 },
    { id: '3', name: 'Mike Williams', party: 'Independent', votes: 0 }
  ]);
  
  const [voters, setVoters] = useState<Voter[]>([]);
  const [votes, setVotes] = useState<Vote[]>([]);
  const [currentVoter, setCurrentVoter] = useState<Voter | null>(null);
  const [electionName, setElectionName] = useState<string>('General Election 2024');

  // Load data from localStorage on mount
  useEffect(() => {
    const savedCandidates = localStorage.getItem('voting-candidates');
    const savedVoters = localStorage.getItem('voting-voters');
    const savedVotes = localStorage.getItem('voting-votes');
    const savedElectionName = localStorage.getItem('election-name');
    
    if (savedCandidates) {
      setCandidates(JSON.parse(savedCandidates));
    }
    
    if (savedVoters) {
      setVoters(JSON.parse(savedVoters));
    }
    
    if (savedVotes) {
      setVotes(JSON.parse(savedVotes));
    }
    
    if (savedElectionName) {
      setElectionName(savedElectionName);
    }
  }, []);

  // Save to localStorage whenever data changes
  useEffect(() => {
    localStorage.setItem('voting-candidates', JSON.stringify(candidates));
  }, [candidates]);

  useEffect(() => {
    localStorage.setItem('voting-voters', JSON.stringify(voters));
  }, [voters]);

  useEffect(() => {
    localStorage.setItem('voting-votes', JSON.stringify(votes));
  }, [votes]);

  useEffect(() => {
    localStorage.setItem('election-name', electionName);
  }, [electionName]);

  const addCandidate = (candidate: Omit<Candidate, "votes">) => {
    const newCandidate: Candidate = {
      ...candidate,
      votes: 0
    };
    setCandidates(prev => [...prev, newCandidate]);
  };

  const removeCandidate = (id: string) => {
    setCandidates(prev => prev.filter(candidate => candidate.id !== id));
  };

  const addVoter = (voterData: Omit<Voter, "id">) => {
    const newVoter: Voter = {
      id: Date.now().toString(),
      ...voterData
    };
    setVoters(prev => [...prev, newVoter]);
  };

  const vote = (candidateId: string) => {
    setCandidates(prev => 
      prev.map(candidate => 
        candidate.id === candidateId 
          ? { ...candidate, votes: candidate.votes + 1 }
          : candidate
      )
    );
  };

  const castVote = (candidateId: string): boolean => {
    if (!currentVoter) return false;
    
    // Check if voter has already voted
    const hasVoted = votes.some(vote => vote.voterName === currentVoter.name);
    if (hasVoted) return false;
    
    // Record the vote
    const newVote: Vote = {
      id: Date.now().toString(),
      voterName: currentVoter.name,
      candidateId,
      timestamp: new Date().toISOString()
    };
    
    setVotes(prev => [...prev, newVote]);
    vote(candidateId);
    
    return true;
  };

  const getResults = (): CandidateResult[] => {
    const totalVotes = getTotalVotes();
    
    return candidates.map(candidate => ({
      ...candidate,
      percentage: totalVotes > 0 ? (candidate.votes / totalVotes) * 100 : 0
    }));
  };

  const getTotalVotes = (): number => {
    return candidates.reduce((total, candidate) => total + candidate.votes, 0);
  };

  const clearAllData = () => {
    setCandidates([]);
    setVoters([]);
    setVotes([]);
    setCurrentVoter(null);
    localStorage.removeItem('voting-candidates');
    localStorage.removeItem('voting-voters');
    localStorage.removeItem('voting-votes');
    localStorage.removeItem('election-name');
    setElectionName('General Election 2024');
  };

  const value: VotingContextType = {
    candidates,
    voters,
    votes,
    currentVoter,
    electionName,
    addCandidate,
    removeCandidate,
    vote,
    addVoter,
    setCurrentVoter,
    castVote,
    getResults,
    getTotalVotes,
    clearAllData,
    setElectionName
  };

  return (
    <VotingContext.Provider value={value}>
      {children}
    </VotingContext.Provider>
  );
};
