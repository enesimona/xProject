class Member(object):
    def __init__(self, name, constituency, party, vote):
        self.name = name
        self.constituency = constituency
        self.party = party
        self.vote = vote
        
    def __str__(self):
        return "%s , constituency: %s, party: %s, vote: %d" % (self.name, self.constituency, self.party, self.vote)