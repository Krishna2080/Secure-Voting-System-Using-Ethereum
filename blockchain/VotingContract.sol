
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

contract SecureVoting {
    struct Vote {
        string voterName;
        string candidateId;
        uint256 timestamp;
        bool exists;
    }
    
    mapping(string => Vote) public votes; // voterName => Vote
    mapping(string => uint256) public candidateVotes; // candidateId => vote count
    mapping(string => bool) public hasVoted; // voterName => voted status
    
    string[] public candidates;
    address public admin;
    bool public votingActive;
    uint256 public totalVotes;
    
    event VoteCast(string indexed voterName, string indexed candidateId, uint256 timestamp, bytes32 txHash);
    event VotingStatusChanged(bool active);
    
    modifier onlyAdmin() {
        require(msg.sender == admin, "Only admin can perform this action");
        _;
    }
    
    modifier votingIsActive() {
        require(votingActive, "Voting is not active");
        _;
    }
    
    constructor(string[] memory _candidates) {
        admin = msg.sender;
        candidates = _candidates;
        votingActive = true;
        
        // Initialize candidate vote counts
        for (uint i = 0; i < _candidates.length; i++) {
            candidateVotes[_candidates[i]] = 0;
        }
    }
    
    function castVote(string memory _voterName, string memory _candidateId) 
        public 
        votingIsActive 
        returns (bool success) 
    {
        require(!hasVoted[_voterName], "Voter has already cast their vote");
        require(isValidCandidate(_candidateId), "Invalid candidate ID");
        require(bytes(_voterName).length > 0, "Voter name cannot be empty");
        
        // Record the vote
        votes[_voterName] = Vote({
            voterName: _voterName,
            candidateId: _candidateId,
            timestamp: block.timestamp,
            exists: true
        });
        
        // Update vote counts
        candidateVotes[_candidateId]++;
        hasVoted[_voterName] = true;
        totalVotes++;
        
        // Emit event with transaction hash
        bytes32 txHash = keccak256(abi.encodePacked(_voterName, _candidateId, block.timestamp));
        emit VoteCast(_voterName, _candidateId, block.timestamp, txHash);
        
        return true;
    }
    
    function isValidCandidate(string memory _candidateId) public view returns (bool) {
        for (uint i = 0; i < candidates.length; i++) {
            if (keccak256(bytes(candidates[i])) == keccak256(bytes(_candidateId))) {
                return true;
            }
        }
        return false;
    }
    
    function getVote(string memory _voterName) public view returns (Vote memory) {
        require(votes[_voterName].exists, "Vote does not exist");
        return votes[_voterName];
    }
    
    function getCandidateVotes(string memory _candidateId) public view returns (uint256) {
        return candidateVotes[_candidateId];
    }
    
    function getAllCandidates() public view returns (string[] memory) {
        return candidates;
    }
    
    function getResults() public view returns (string[] memory, uint256[] memory) {
        uint256[] memory voteCounts = new uint256[](candidates.length);
        
        for (uint i = 0; i < candidates.length; i++) {
            voteCounts[i] = candidateVotes[candidates[i]];
        }
        
        return (candidates, voteCounts);
    }
    
    function toggleVoting() public onlyAdmin {
        votingActive = !votingActive;
        emit VotingStatusChanged(votingActive);
    }
    
    function addCandidate(string memory _candidateId) public onlyAdmin {
        require(!votingActive, "Cannot add candidates while voting is active");
        candidates.push(_candidateId);
        candidateVotes[_candidateId] = 0;
    }
}
